'''
Regime-switching macro state-space model.

Regime transition (Markov, optionally covariate-dependent):
    P(s_t = j | z_{t-1}, s_{t-1} = i) = softmax(a_ij + b_ij z_{t-1})_j

Latent state  z_t = [x_t, g_t*, u_t*, pi_t^e, r_t*, x_{t-1}]' in R^6:
    x_t      — output gap
    g_t*     — potential GDP growth
    u_t*     — natural unemployment rate
    pi_t^e   — expected inflation
    r_t*     — neutral real interest rate
    x_{t-1}  — lagged output gap (augmentation to keep H constant)

Transition:
    z_t = A z_{t-1} + a + eps_t,   eps_t ~ N(0, Q_{s_t})

    A = [[rho_x, 0,     0,     0,       -lambda_r, 0],
         [0,     rho_g, 0,     0,        0,         0],
         [0,     0,     rho_u, 0,        0,         0],
         [0,     0,     0,     rho_pi,   0,         0],
         [0,     0,     0,     0,        rho_r,     0],
         [1,     0,     0,     0,        0,         0]]

    a = [0, (1-rho_g)g_bar, (1-rho_u)u_bar, (1-rho_pi)pi_bar, (1-rho_r)r_bar, 0]'

    Q_{s_t} = diag(sigma_{s_t}^2, 0) — last component deterministic

Observation  y_t = [DeltaY_t, u_t, pi_t, i_t]' in R^4:
    y_t = H z_t + b_t + eta_t,   eta_t ~ N(0, R)

    H = [[1,       1, 0, 0,                    0,           -1],
         [-beta_u, 0, 1, 0,                    0,            0],
         [kappa,   0, 0, 1,                    0,            0],
         [(1-r_i)phi_x, 0, 0, (1-r_i)(1+phi_pi), (1-r_i),  0]]

    b_t = [0, 0, 0, rho_i i_{t-1} - (1-rho_i) phi_pi pi*]'   (time-varying)
'''

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.stats import multivariate_normal

from models.base import StateSpaceModel
from utils import row_softmax, symmetrize


# ── dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ModelDims:
    n_regimes: int
    n_state: int = 6         # augmented latent state dimension
    n_obs: int = 4           # number of observed macro series
    n_covariates: int = 0    # covariates entering regime transition probs


@dataclass
class RegimeStructure:
    """Controls which parameters vary across regimes vs. are shared."""
    regime_specific_a: bool = False
    regime_specific_a_intercept: bool = False
    regime_specific_q: bool = True   # volatility is regime-specific by default
    regime_specific_h: bool = False
    regime_specific_b: bool = False
    regime_specific_r: bool = False


@dataclass
class MacroParams:
    """
    Structural macro parameters.

    State vector: z_t = [x_t, g_t*, u_t*, pi_t^e, r_t*, x_{t-1}]
    Observed:     y_t = [GDP_growth, unemployment, inflation, nominal_rate]
    """
    # Persistence
    rho_x: float
    rho_g: float
    rho_u: float
    rho_pi: float
    rho_r: float

    # Long-run means
    g_bar: float
    u_bar: float
    pi_bar: float
    r_bar: float

    # Structural slopes
    lambda_r: float    # IS curve: rate sensitivity of output gap
    beta_u: float      # Okun's law slope
    kappa: float       # Phillips curve slope

    # Taylor rule
    rho_i: float       # interest rate smoothing
    phi_pi: float      # inflation response
    phi_x: float       # output gap response
    pi_target: float   # inflation target

    # State shock std devs, shape (K, 5) — per-regime, pre-augmentation
    state_sigmas: np.ndarray

    # Measurement shock std devs, shape (4,)
    obs_sigmas: np.ndarray

    # Regime transition intercepts, shape (K, K)
    trans_intercepts: np.ndarray

    # Regime transition covariate coefficients, shape (K, K, n_covariates) or None
    trans_coefs: Optional[np.ndarray] = None

    # Initial distribution (None → diffuse defaults)
    init_state_mean: Optional[np.ndarray] = None
    init_state_cov: Optional[np.ndarray] = None
    init_regime_probs: Optional[np.ndarray] = None


# ── model ─────────────────────────────────────────────────────────────────────

class RegimeSwitchingMacro(StateSpaceModel):
    def __init__(
        self,
        params: MacroParams,
        dims: ModelDims,
        structure: Optional[RegimeStructure] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(seed=seed, state_dim=dims.n_state, obs_dim=dims.n_obs)
        self.macro_params = params
        self.dims = dims
        self.structure = structure or RegimeStructure()
        self.n_regimes = dims.n_regimes
        self.rng = np.random.default_rng(seed)
        self._validate_params()

        self.regime_probs_stationary = (
            params.init_regime_probs
            if params.init_regime_probs is not None
            else self._solve_stationary_regime_dist()
        )

        p = params
        self.macro_params_dict = {
            'rho_x': p.rho_x, 'rho_g': p.rho_g, 'rho_u': p.rho_u,
            'rho_pi': p.rho_pi, 'rho_r': p.rho_r,
            'g_bar': p.g_bar, 'u_bar': p.u_bar, 'pi_bar': p.pi_bar, 'r_bar': p.r_bar,
            'lambda_r': p.lambda_r, 'beta_u': p.beta_u, 'kappa': p.kappa,
            'rho_i': p.rho_i, 'phi_pi': p.phi_pi, 'phi_x': p.phi_x,
            'pi_target': p.pi_target,
        }

    def __repr__(self):
        return (
            f"RegimeSwitchingMacro("
            f"n_regimes={self.n_regimes}, "
            f"state_dim={self.state_dim}, obs_dim={self.obs_dim})"
        )

    def describe(self):
        p = self.macro_params
        return (
            f"{self.__class__.__name__}\n"
            f"  Regime-switching macro state-space model\n"
            f"  Regimes  : {self.n_regimes}  (regime-specific Q, shared A/H/R)\n"
            f"  State    : z_t = [x_t, g_t*, u_t*, pi_t^e, r_t*, x_{{t-1}}]  (dim=6)\n"
            f"  Observed : y_t = [GDP growth, unemployment, inflation, nominal rate]  (dim=4)\n"
            f"  Transition:  z_t = A z_{{t-1}} + a + eps_t,  eps_t ~ N(0, Q_{{s_t}})\n"
            f"  Observation: y_t = H z_t + b_t(i_{{t-1}}) + eta_t,  eta_t ~ N(0, R)\n"
            f"  Scalar parameters: {self.macro_params_dict}\n"
            f"  State sigmas (per regime):\n{p.state_sigmas}\n"
            f"  Obs sigmas: {p.obs_sigmas}"
        )

    def update_params(self, constrained_params: 'MacroParams'):
        """Update model from a constrained MacroParams object."""
        self.macro_params = constrained_params
        p = constrained_params
        self.macro_params_dict = {
            'rho_x': p.rho_x, 'rho_g': p.rho_g, 'rho_u': p.rho_u,
            'rho_pi': p.rho_pi, 'rho_r': p.rho_r,
            'g_bar': p.g_bar, 'u_bar': p.u_bar, 'pi_bar': p.pi_bar, 'r_bar': p.r_bar,
            'lambda_r': p.lambda_r, 'beta_u': p.beta_u, 'kappa': p.kappa,
            'rho_i': p.rho_i, 'phi_pi': p.phi_pi, 'phi_x': p.phi_x,
            'pi_target': p.pi_target,
        }
        if p.init_regime_probs is None:
            self.regime_probs_stationary = self._solve_stationary_regime_dist()
        self.check_params_validity()

    def check_params_validity(self):
        self._validate_params()

    # ── validation ────────────────────────────────────────────────────────────

    def _validate_params(self):
        p, k = self.macro_params, self.n_regimes
        assert p.state_sigmas.shape == (k, 5), f"state_sigmas must be ({k}, 5)"
        assert p.obs_sigmas.shape == (4,), "obs_sigmas must be shape (4,)"
        assert p.trans_intercepts.shape == (k, k), f"trans_intercepts must be ({k}, {k})"
        if p.trans_coefs is not None:
            assert p.trans_coefs.shape == (k, k, self.dims.n_covariates)
        for name in ["rho_x", "rho_g", "rho_u", "rho_pi", "rho_r", "rho_i"]:
            assert abs(getattr(p, name)) < 1.0, f"|{name}| must be < 1"
        assert np.all(p.state_sigmas > 0), "all state_sigmas must be positive"
        assert np.all(p.obs_sigmas > 0), "all obs_sigmas must be positive"

    # ── stationary regime distribution ────────────────────────────────────────

    def _solve_stationary_regime_dist(self) -> np.ndarray:
        """Solve pi P = pi, sum(pi) = 1 for the constant transition matrix."""
        k = self.n_regimes
        p_trans = self.transition_probs(0)       # (K, K) row-stochastic
        a = (np.eye(k) - p_trans).T              # (K, K)
        a[-1] = np.ones(k)                       # replace last row: normalization
        b = np.zeros(k)
        b[-1] = 1.0
        return np.linalg.solve(a, b)

    # ── regime transition probability ─────────────────────────────────────────

    def transition_probs(self, t: int, covariates: Optional[np.ndarray] = None) -> np.ndarray:
        """Return P where P[i, j] = P(s_t = j | s_{t-1} = i)."""
        c = self.macro_params.trans_intercepts.copy()
        if self.macro_params.trans_coefs is not None:
            if covariates is None:
                raise ValueError("covariates required (trans_coefs is set)")
            w = covariates[max(t - 1, 0)]
            c = c + np.einsum("ijk,k->ij", self.macro_params.trans_coefs, w)
        return row_softmax(c)

    # ── matrix builders ───────────────────────────────────────────────────────

    def build_A(self, s: int) -> np.ndarray:
        """6×6 transition matrix (shared across regimes)."""
        p = self.macro_params
        a = np.zeros((6, 6))
        a[0, 0] = p.rho_x
        a[0, 4] = -p.lambda_r
        a[1, 1] = p.rho_g
        a[2, 2] = p.rho_u
        a[3, 3] = p.rho_pi
        a[4, 4] = p.rho_r
        a[5, 0] = 1.0   # x_{t-1}^{new} = x_t^{old}
        return a

    def build_intercept(self, s: int) -> np.ndarray:
        """6-vector state intercept a (shared across regimes)."""
        p = self.macro_params
        a = np.zeros(6)
        a[1] = (1.0 - p.rho_g) * p.g_bar
        a[2] = (1.0 - p.rho_u) * p.u_bar
        a[3] = (1.0 - p.rho_pi) * p.pi_bar
        a[4] = (1.0 - p.rho_r) * p.r_bar
        return a

    def build_Q(self, s: int) -> np.ndarray:
        """6×6 state covariance — regime-specific; last component is deterministic."""
        sig = self.macro_params.state_sigmas[s]
        q = np.zeros((6, 6))
        q[:5, :5] = np.diag(sig ** 2)
        return q

    def build_H(self, s: int) -> np.ndarray:
        """4×6 observation matrix (shared across regimes)."""
        p = self.macro_params
        h = np.zeros((4, 6))
        # GDP growth: g_t* + x_t - x_{t-1}
        h[0, 0] = 1.0;  h[0, 1] = 1.0;  h[0, 5] = -1.0
        # Unemployment: u_t* - beta_u x_t
        h[1, 0] = -p.beta_u;  h[1, 2] = 1.0
        # Inflation: pi_t^e + kappa x_t
        h[2, 0] = p.kappa;  h[2, 3] = 1.0
        # Nominal rate (Taylor rule, linearised around pi*)
        h[3, 0] = (1.0 - p.rho_i) * p.phi_x
        h[3, 3] = (1.0 - p.rho_i) * (1.0 + p.phi_pi)
        h[3, 4] = (1.0 - p.rho_i)
        return h

    def build_b(self, s: int, i_lag: float) -> np.ndarray:
        """
        4-vector observation intercept.  Only the interest-rate row is non-zero;
        it depends on the lagged nominal rate i_{t-1}.
        """
        p = self.macro_params
        b = np.zeros(4)
        b[3] = p.rho_i * i_lag - (1.0 - p.rho_i) * p.phi_pi * p.pi_target
        return b

    def build_R(self, s: int) -> np.ndarray:
        """4×4 measurement noise covariance (shared across regimes)."""
        return np.diag(self.macro_params.obs_sigmas ** 2)

    # ── SSM interface ─────────────────────────────────────────────────────────

    def sample_initial_distribution(self):
        """
        Sample (z_0, s_0) from the initial distribution.

        s_0 ~ Categorical(regime_probs_stationary)
        z_0 ~ N(init_state_mean, init_state_cov),  z_0[5] set to 0.

        Returns
        -------
        z_0 : (6,) initial latent state
        s_0 : int  initial regime
        """
        p, n, k = self.macro_params, self.dims.n_state, self.n_regimes
        s_0 = self.rng.choice(k, p=self.regime_probs_stationary)
        m0 = p.init_state_mean if p.init_state_mean is not None else np.zeros(n)
        p0 = p.init_state_cov  if p.init_state_cov  is not None else np.eye(n)
        l = np.linalg.cholesky(p0 + 1e-10 * np.eye(n))
        z_0 = m0 + l @ self.rng.standard_normal(n)
        z_0[5] = 0.0   # x_{-1} initialised to zero
        return z_0, s_0

    def initial_density(self, z, s: int) -> float:
        """p(z_0, s_0=s) = pi_0[s] * N(z | m0, P0)."""
        p, n = self.macro_params, self.dims.n_state
        m0 = p.init_state_mean if p.init_state_mean is not None else np.zeros(n)
        p0 = p.init_state_cov  if p.init_state_cov  is not None else np.eye(n)
        return self.regime_probs_stationary[s] * multivariate_normal.pdf(z, mean=m0, cov=p0)

    def transition(self, z_prev, regime_prev: int, t: int = 0, covariates=None):
        """
        Sample (z_t, s_t) given z_{t-1} and s_{t-1}.

        Parameters
        ----------
        z_prev      : (6,) previous latent state
        regime_prev : int  previous regime s_{t-1}
        t           : time index (used for covariate look-up)
        covariates  : optional array for time-varying transition probs

        Returns
        -------
        z_next : (6,) new latent state
        s_next : int  new regime
        """
        k, n = self.n_regimes, self.dims.n_state
        p_trans = self.transition_probs(t, covariates)
        s_next = self.rng.choice(k, p=p_trans[regime_prev])

        eps = np.zeros(n)
        eps[:5] = self.macro_params.state_sigmas[s_next] * self.rng.standard_normal(5)
        z_next = self.build_A(s_next) @ z_prev + self.build_intercept(s_next) + eps
        return z_next, s_next

    def observation(self, z, regime: int, i_lag: float = 0.0):
        """
        Sample y_t given z_t, s_t, and the lagged nominal rate i_{t-1}.

        Returns
        -------
        y : (4,) [GDP growth, unemployment, inflation, nominal rate]
        """
        h = self.build_H(regime)
        b = self.build_b(regime, i_lag)
        r = self.build_R(regime)
        l_r = np.linalg.cholesky(r)
        return h @ z + b + l_r @ self.rng.standard_normal(self.dims.n_obs)

    def log_transition_density(self, z_next, z_prev, regime: int) -> float:
        """
        Log p(z_t | z_{t-1}, s_t=regime).

        The last component of z_t is deterministic (x_{t-1} = x_{t-1}), so
        we return the log-density of the non-degenerate first 5 components only.
        """
        mean_full = self.build_A(regime) @ z_prev + self.build_intercept(regime)
        q5 = self.build_Q(regime)[:5, :5]
        return multivariate_normal.logpdf(z_next[:5], mean=mean_full[:5], cov=q5)

    def log_observation_density(self, y, z, regime: int, i_lag: float = 0.0) -> float:
        """Log p(y_t | z_t, s_t=regime, i_{t-1})."""
        h = self.build_H(regime)
        b = self.build_b(regime, i_lag)
        r = self.build_R(regime)
        return multivariate_normal.logpdf(y, mean=h @ z + b, cov=r)

    # ── data simulation ───────────────────────────────────────────────────────

    def generate_data(self, t: int, covariates=None):
        """
        Simulate a synthetic path from the model.

        Returns
        -------
        states        : (T, 6)   augmented latent state z_t
        observations  : (T, 4)   observed macro series y_t
        regimes       : (T,) int regime s_t
        log_likelihood: float    sum_t log p(y_t | z_{1:t-1}, y_{1:t-1})
        """
        n, m = self.dims.n_state, self.dims.n_obs
        states        = np.zeros((t, n))
        observations  = np.zeros((t, m))
        regimes       = np.zeros(t, dtype=int)
        log_likelihood = 0.0

        z0, s0 = self.sample_initial_distribution()
        states[0]       = z0
        regimes[0]      = s0
        observations[0] = self.observation(z0, s0, i_lag=0.0)
        log_likelihood += self.log_observation_density(observations[0], z0, s0, i_lag=0.0)

        for step in range(1, t):
            z_new, s_new = self.transition(
                states[step - 1], regimes[step - 1], t=step, covariates=covariates
            )
            i_lag = observations[step - 1, 3]
            states[step]       = z_new
            regimes[step]      = s_new
            observations[step] = self.observation(z_new, s_new, i_lag=i_lag)
            log_likelihood    += self.log_observation_density(
                observations[step], z_new, s_new, i_lag=i_lag
            )

        return states, observations, regimes, log_likelihood

    # ── parameter transforms ──────────────────────────────────────────────────

    def constrain_params(self, unconstrained_params) -> MacroParams:
        """
        Map an unconstrained flat vector → MacroParams.

        Layout (total length = 16 + K*5 + 4 + K*K):
            atanh(rho_x), atanh(rho_g), atanh(rho_u), atanh(rho_pi), atanh(rho_r),
            g_bar, u_bar, pi_bar, r_bar,
            lambda_r, beta_u, kappa,
            atanh(rho_i), phi_pi, phi_x, pi_target,
            log(state_sigmas) — K×5 row-major,
            log(obs_sigmas)   — 4,
            trans_intercepts  — K×K row-major (unconstrained)
        """
        k = self.n_regimes
        z = np.asarray(unconstrained_params, dtype=float)
        idx = 0

        def pull(n):
            nonlocal idx
            v = z[idx:idx + n]; idx += n; return v

        rhos   = np.tanh(pull(5))
        means  = pull(4)
        slopes = pull(3)
        taylor = pull(4)

        state_sigmas = np.exp(pull(k * 5)).reshape(k, 5)
        obs_sigmas   = np.exp(pull(4))
        trans_int    = pull(k * k).reshape(k, k)

        p = self.macro_params
        return MacroParams(
            rho_x=rhos[0], rho_g=rhos[1], rho_u=rhos[2], rho_pi=rhos[3], rho_r=rhos[4],
            g_bar=means[0], u_bar=means[1], pi_bar=means[2], r_bar=means[3],
            lambda_r=slopes[0], beta_u=slopes[1], kappa=slopes[2],
            rho_i=np.tanh(taylor[0]), phi_pi=taylor[1], phi_x=taylor[2], pi_target=taylor[3],
            state_sigmas=state_sigmas,
            obs_sigmas=obs_sigmas,
            trans_intercepts=trans_int,
            trans_coefs=p.trans_coefs,
            init_state_mean=p.init_state_mean,
            init_state_cov=p.init_state_cov,
            init_regime_probs=p.init_regime_probs,
        )

    def unconstrain_params(self, constrained_params: MacroParams) -> np.ndarray:
        """Inverse of constrain_params."""
        p = constrained_params
        return np.concatenate([
            np.arctanh([p.rho_x, p.rho_g, p.rho_u, p.rho_pi, p.rho_r]),
            [p.g_bar, p.u_bar, p.pi_bar, p.r_bar],
            [p.lambda_r, p.beta_u, p.kappa],
            [np.arctanh(p.rho_i), p.phi_pi, p.phi_x, p.pi_target],
            np.log(p.state_sigmas).flatten(),
            np.log(p.obs_sigmas),
            p.trans_intercepts.flatten(),
        ])
