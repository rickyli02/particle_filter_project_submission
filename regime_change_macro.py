# Imports

'''
The code implements the following regime change model:

\begin{cases}
\mathbb{P}( s_{t} = j \,\mid\, z_{t-1}, s_{t-1} = i ) =  \text{sigmoid}(a_{ij} + b_{ij}z_{t-1}),\quad i,j \in \{1, ... , N_{regime}\} \\
z_t = A_{s_{t}} z_{t-1} + a_{s_{t}} +\sigma_{s_{t}}^\top \epsilon_t\\
y_t = b_{s_{t}} + H_{s_{t}} z_{t} + \tau^\top \eta_t \\
\epsilon_t \sim \mathcal{N}_{d}(0, 1) \\
\eta_t \sim \mathcal{N}_{m}(0, 1)
\end{cases}

where the observables are

- real GDP growth rate
- observed unemployment rate
- observed inflation rate
- observed short term nominal interest rate

the latent state represents the following

- regimes 
    - model should impose some structure
    - ideally 2-4 regimes to avoid overfitting and to maintain interpretability
    - for example, regime 1 is a "high-volatility" regime, regime 2 is a "medium-volatility" regime, and regime 3 is a "low-volatility" regime
    - alternatively, a combination of high / low volatility and high / low growth regimes could be considered
    - some parameters could be shared across regimes; affects interpretation
- output gap: log real GDP minus log potential GDP
- growth rate of potential GDP
- natural rate of unemployment
- expected inflation
- neutral real interest rate

the ex-ante real interest rate is defined as observed short term nominal interest rate  - expected inflation

Conditional on the regime $s_t$, the state space model is given by the following: 
The transition matrix $A_{s_t}$ follows the following form:
$$
A
=
\begin{bmatrix}
\rho_x & 0      & 0      & 0        & -\lambda_r \\
0      & \rho_g & 0      & 0        & 0          \\
0      & 0      & \rho_u & 0        & 0          \\
0      & 0      & 0      & \rho_\pi & 0          \\
0      & 0      & 0      & 0        & \rho_r
\end{bmatrix}.
$$

The intercept vector is

$$
a
=
\begin{bmatrix}
0 \\
(1-\rho_g)\bar{g} \\
(1-\rho_u)\bar{u} \\
(1-\rho_\pi)\bar{\pi} \\
(1-\rho_r)\bar{r}
\end{bmatrix}.
$$

The state covariance matrix is

$$
Q
=
\begin{bmatrix}
\sigma_x^2        & 0                  & 0                      & 0                         & 0 \\
0                 & \sigma_g^2         & 0                      & 0                         & 0 \\
0                 & 0                  & (\sigma_u^*)^2          & 0                         & 0 \\
0                 & 0                  & 0                      & (\sigma_\pi^e)^2           & 0 \\
0                 & 0                  & 0                      & 0                         & (\sigma_r^*)^2
\end{bmatrix}.
$$

The first row says that the output gap is persistent and may be negatively affected by a higher neutral-rate-adjusted real-rate component.

The observed macroeconomic variables satisfy

$$
y_t = H_t z_t + b_t + \varepsilon_t,
\qquad
\varepsilon_t \sim \mathcal{N}(0,R).
$$

The measurement equations are

$$
\Delta Y_t
=
g_t^* + x_t - x_{t-1} + \varepsilon_t^Y,
$$

$$
u_t
=
u_t^* - \beta_u x_t + \varepsilon_t^u,
$$

$$
\pi_t
=
\pi_t^e + \kappa x_t + \varepsilon_t^\pi,
$$

$$
i_t
=
\rho_i i_{t-1}
+
(1-\rho_i)
\left[
r_t^*
+
\pi_t^e
+
\phi_\pi(\pi_t^e-\pi^*)
+
\phi_x x_t
\right]
+
\varepsilon_t^i.
$$

Because the GDP growth equation contains $x_{t-1}$ and the interest-rate equation contains $i_{t-1}$, the observation equation can be written as a time-varying observation equation:

$$
y_t = H_t z_t + b_t + \varepsilon_t.
$$

To avoid a time-varying $b_t$ in the first row, the state is augmented with the lagged output gap:
$z_t = [x_t,\, g_t^*,\, u_t^*,\, \pi_t^e,\, r_t^*,\, x_{t-1}]^\top \in \mathbb{R}^6$.
This gives a constant $H$ and a $b_t$ that depends only on the lagged interest rate.

$$
H
=
\begin{bmatrix}
1              & 1 & 0 & 0                         & 0            & -1 \\
-\beta_u       & 0 & 1 & 0                         & 0            & 0  \\
\kappa         & 0 & 0 & 1                         & 0            & 0  \\
(1-\rho_i)\phi_x
               & 0 & 0 & (1-\rho_i)(1+\phi_\pi)   & (1-\rho_i)   & 0
\end{bmatrix},
$$

and

$$
b_t
=
\begin{bmatrix}
0 \\
0 \\
0 \\
\rho_i i_{t-1} - (1-\rho_i)\phi_\pi \pi^*
\end{bmatrix}.
$$

The measurement covariance matrix is

$$
R
=
\begin{bmatrix}
\sigma_Y^2 & 0          & 0             & 0 \\
0          & \sigma_u^2 & 0             & 0 \\
0          & 0          & \sigma_\pi^2  & 0 \\
0          & 0          & 0             & \sigma_i^2
\end{bmatrix}.
$$

'''

# Parameter dataclass
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np


@dataclass
class ModelDims:
    """
    Dimensions of the regime-switching macro state-space model.
    """
    n_regimes: int
    n_state: int          # latent state dimension, including augmentation
    n_obs: int            # number of observed variables
    n_covariates: int = 0 # covariates entering transition probabilities


@dataclass
class RegimeStructure:
    """
    Controls which parameters are regime-specific and which are shared.

    Example:
    - regime_specific_A = False: same transition matrix across regimes
    - regime_specific_Q = True: regime-dependent volatility
    - regime_specific_H = False: same observation matrix across regimes
    """
    regime_specific_A: bool = False
    regime_specific_a: bool = False
    regime_specific_Q: bool = True
    regime_specific_H: bool = False
    regime_specific_b: bool = False
    regime_specific_R: bool = False


@dataclass
class MacroParams:
    """
    Structural macro parameters.

    State vector:
        alpha_t = [x_t, g_t_star, u_t_star, pi_t_e, r_t_star, x_lag_t]

    Observed vector:
        y_t = [gdp_growth_t, unemployment_t, inflation_t, nominal_rate_t]
    """

    # Persistence parameters
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
    lambda_r: float
    beta_u: float
    kappa: float

    # Taylor-rule parameters
    rho_i: float
    phi_pi: float
    phi_x: float
    pi_target: float

    # State shock standard deviations by regime
    # Shape: (K, 5), before augmentation.
    state_sigmas: np.ndarray

    # Measurement shock standard deviations
    # Shape: (4,)
    obs_sigmas: np.ndarray

    # Transition probability parameters
    # Shape: intercepts (K, K), covariate coeffs (K, K, n_covariates)
    trans_intercepts: np.ndarray
    trans_coefs: Optional[np.ndarray] = None

    # Initial distributions
    init_state_mean: Optional[np.ndarray] = None
    init_state_cov: Optional[np.ndarray] = None
    init_regime_probs: Optional[np.ndarray] = None


# Softmax
def row_softmax(X: np.ndarray) -> np.ndarray:
    """
    Numerically stable row-wise softmax.
    """
    X_shifted = X - np.max(X, axis=1, keepdims=True)
    E = np.exp(X_shifted)
    return E / np.sum(E, axis=1, keepdims=True)


def symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def kalman_predict(m, P, A, a, Q):
    """
    State prediction:
        alpha_t | y_{1:t-1}
    """
    m_pred = A @ m + a
    P_pred = A @ P @ A.T + Q
    P_pred = symmetrize(P_pred)
    return m_pred, P_pred


def kalman_update(m_pred, P_pred, y_t, H, b, R):
    """
    Observation update:
        alpha_t | y_{1:t}
    Also returns log p(y_t | y_{1:t-1}).
    """
    from scipy.linalg import cho_factor, cho_solve

    obs_pred = H @ m_pred + b
    innovation = y_t - obs_pred

    S = H @ P_pred @ H.T + R
    S = symmetrize(S)

    c, lower = cho_factor(S, lower=True, check_finite=False)
    S_inv_innov = cho_solve((c, lower), innovation, check_finite=False)

    logdet_S = 2.0 * np.sum(np.log(np.diag(c)))
    n_obs = len(y_t)

    loglik = -0.5 * (
        n_obs * np.log(2.0 * np.pi)
        + logdet_S
        + innovation.T @ S_inv_innov
    )

    K_gain = P_pred @ H.T
    K_gain = cho_solve((c, lower), K_gain.T, check_finite=False).T

    m_filt = m_pred + K_gain @ innovation
    P_filt = P_pred - K_gain @ H @ P_pred
    P_filt = symmetrize(P_filt)

    return m_filt, P_filt, loglik, innovation, S


class RegimeMacroModel:
    def __init__(
        self,
        params: MacroParams,
        dims: ModelDims,
        structure: Optional[RegimeStructure] = None,
    ):
        self.params = params
        self.dims = dims
        self.structure = structure or RegimeStructure()
        self.validate_params(params, dims)
    
    def validate_params(self, params: MacroParams, dims: ModelDims) -> None:
        K = dims.n_regimes

        assert params.state_sigmas.shape == (K, 5), (
            f"state_sigmas should have shape {(K, 5)}, "
            f"got {params.state_sigmas.shape}"
        )

        assert params.obs_sigmas.shape == (4,), (
            f"obs_sigmas should have shape (4,), got {params.obs_sigmas.shape}"
        )

        assert params.trans_intercepts.shape == (K, K), (
            f"trans_intercepts should have shape {(K, K)}, "
            f"got {params.trans_intercepts.shape}"
        )

        if params.trans_coefs is not None:
            assert params.trans_coefs.shape == (K, K, dims.n_covariates), (
                f"trans_coefs should have shape {(K, K, dims.n_covariates)}, "
                f"got {params.trans_coefs.shape}"
            )

        for name in ["rho_x", "rho_g", "rho_u", "rho_pi", "rho_r", "rho_i"]:
            value = getattr(params, name)
            assert abs(value) < 1.0, f"{name} should satisfy abs({name}) < 1"

        assert np.all(params.state_sigmas > 0), "All state sigmas must be positive"
        assert np.all(params.obs_sigmas > 0), "All observation sigmas must be positive"

    def transition_probs(self, t: int, covariates: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns P_t where P_t[i, j] = P(s_t = j | s_{t-1} = i, covariates_{t-1}).

        If no covariates are used, this is a fixed transition matrix.
        """
        C = self.params.trans_intercepts.copy()

        if self.params.trans_coefs is not None:
            if covariates is None:
                raise ValueError("Covariates required because trans_coefs is not None.")

            w = covariates[max(t - 1, 0)]
            C = C + np.einsum("ijk,k->ij", self.params.trans_coefs, w)

        return row_softmax(C)

    def A(self, s: int) -> np.ndarray:
        """
        Augmented transition matrix for:
            [x_t, g_t*, u_t*, pi_t^e, r_t*, x_{t-1}]

        The last row copies previous x_t into lagged x for the next period.
        """
        p = self.params

        A = np.zeros((6, 6))

        # Core 5-dimensional latent state transition
        A[0, 0] = p.rho_x
        A[0, 4] = -p.lambda_r

        A[1, 1] = p.rho_g
        A[2, 2] = p.rho_u
        A[3, 3] = p.rho_pi
        A[4, 4] = p.rho_r

        # Lag update: x_{t-1}^{new} = x_t^{old}
        A[5, 0] = 1.0

        return A

    def a(self, s: int) -> np.ndarray:
        """
        Augmented intercept vector.
        """
        p = self.params

        a = np.zeros(6)
        a[0] = 0.0
        a[1] = (1.0 - p.rho_g) * p.g_bar
        a[2] = (1.0 - p.rho_u) * p.u_bar
        a[3] = (1.0 - p.rho_pi) * p.pi_bar
        a[4] = (1.0 - p.rho_r) * p.r_bar
        a[5] = 0.0

        return a

    def Q(self, s: int) -> np.ndarray:
        """
        Regime-specific state covariance for augmented state.

        The lagged x component has zero innovation variance because it is deterministic:
            x_lag_t = x_{t-1}.
        """
        sig = self.params.state_sigmas[s]

        Q = np.zeros((6, 6))
        Q[:5, :5] = np.diag(sig ** 2)
        Q[5, 5] = 0.0

        return Q

    def H(self, s: int, t: int, y_lag: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Observation matrix for:
            y_t = [GDP growth, unemployment, inflation, nominal rate]

        Uses Taylor rule based on latent expected inflation, not observed inflation.
        """
        p = self.params

        H = np.zeros((4, 6))

        # GDP growth: g_t* + x_t - x_{t-1}
        H[0, 0] = 1.0
        H[0, 1] = 1.0
        H[0, 5] = -1.0

        # Unemployment: u_t = u_t* - beta_u x_t + noise
        H[1, 0] = -p.beta_u
        H[1, 2] = 1.0

        # Inflation: pi_t = pi_t^e + kappa x_t + noise
        H[2, 0] = p.kappa
        H[2, 3] = 1.0

        # Interest rate:
        # i_t = rho_i i_{t-1}
        #     + (1-rho_i)[r_t* + pi_t^e + phi_pi(pi_t^e - pi*) + phi_x x_t]
        #     + noise
        H[3, 0] = (1.0 - p.rho_i) * p.phi_x
        H[3, 3] = (1.0 - p.rho_i) * (1.0 + p.phi_pi)
        H[3, 4] = (1.0 - p.rho_i)

        return H

    def b(self, s: int, t: int, y: np.ndarray) -> np.ndarray:
        """
        Observation intercept.

        y[t] = [GDP growth, unemployment, inflation, nominal interest rate]

        The only time-varying intercept here comes from lagged nominal rate.
        """
        p = self.params

        b = np.zeros(4)

        if t == 0:
            i_lag = y[0, 3]
        else:
            i_lag = y[t - 1, 3]

        b[3] = p.rho_i * i_lag - (1.0 - p.rho_i) * p.phi_pi * p.pi_target

        return b

    def R(self, s: int) -> np.ndarray:
        """
        Measurement covariance.
        """
        return np.diag(self.params.obs_sigmas ** 2)





# Parameter estimation using Kim's method
@dataclass
class KimFilterResult:
    loglik: float
    filtered_means: np.ndarray          # shape (T, K, n_state)
    filtered_covs: np.ndarray           # shape (T, K, n_state, n_state)
    regime_probs: np.ndarray            # shape (T, K)  — P(s_t=j | y_{1:t})
    predicted_regime_probs: np.ndarray  # shape (T, K)  — P(s_t=j | y_{1:t-1})
    state_mean: np.ndarray              # shape (T, n_state) — regime-weighted collapsed mean
    state_cov: np.ndarray               # shape (T, n_state, n_state)
    innovations: Optional[np.ndarray] = None


def kim_filter(
    model: RegimeMacroModel,
    y: np.ndarray,
    covariates: Optional[np.ndarray] = None,
) -> KimFilterResult:
    """
    Approximate filtering for Markov-switching linear Gaussian state-space model.

    Uses Kim-style collapsing of Gaussian mixtures.
    """
    T = y.shape[0]
    K = model.dims.n_regimes
    n = model.dims.n_state
    m_obs = model.dims.n_obs

    p = model.params

    # Initial state distribution
    if p.init_state_mean is None:
        m0 = np.zeros(n)
    else:
        m0 = p.init_state_mean

    if p.init_state_cov is None:
        P0 = np.eye(n) * 10.0
    else:
        P0 = p.init_state_cov

    if p.init_regime_probs is None:
        regime_probs_prev = np.ones(K) / K
    else:
        regime_probs_prev = p.init_regime_probs

    means_prev = np.repeat(m0[None, :], K, axis=0)
    covs_prev = np.repeat(P0[None, :, :], K, axis=0)

    filtered_means = np.zeros((T, K, n))
    filtered_covs = np.zeros((T, K, n, n))
    regime_probs = np.zeros((T, K))
    predicted_regime_probs = np.zeros((T, K))
    innovations = np.zeros((T, K, m_obs))

    total_loglik = 0.0

    for t in range(T):
        P_trans = model.transition_probs(t, covariates)

        # Pairwise objects indexed by previous regime i and current regime j
        pair_means = np.zeros((K, K, n))
        pair_covs = np.zeros((K, K, n, n))
        pair_loglik = np.zeros((K, K))
        pair_weights_prior = np.zeros((K, K))

        for i in range(K):
            for j in range(K):
                A_j = model.A(j)
                a_j = model.a(j)
                Q_j = model.Q(j)
                H_j = model.H(j, t, y)
                b_j = model.b(j, t, y)
                R_j = model.R(j)

                m_pred, P_pred = kalman_predict(
                    means_prev[i],
                    covs_prev[i],
                    A_j,
                    a_j,
                    Q_j,
                )

                m_filt, P_filt, ll_ij, innov_ij, S_ij = kalman_update(
                    m_pred,
                    P_pred,
                    y[t],
                    H_j,
                    b_j,
                    R_j,
                )

                pair_means[i, j] = m_filt
                pair_covs[i, j] = P_filt
                pair_loglik[i, j] = ll_ij

                pair_weights_prior[i, j] = regime_probs_prev[i] * P_trans[i, j]

        # Compute total likelihood contribution using log-sum-exp
        log_pair_weights = np.log(pair_weights_prior + 1e-300) + pair_loglik
        ll_t = logsumexp_2d(log_pair_weights)
        total_loglik += ll_t

        # Posterior pair weights P(s_{t-1}=i, s_t=j | y_{1:t})
        pair_weights_post = np.exp(log_pair_weights - ll_t)

        # Current regime probabilities P(s_t=j | y_{1:t})
        regime_probs_t = pair_weights_post.sum(axis=0)

        # Predicted current regime probabilities before observing y_t
        predicted_regime_probs[t] = pair_weights_prior.sum(axis=0)

        # Collapse mixture over previous regimes for each current regime j
        means_t = np.zeros((K, n))
        covs_t = np.zeros((K, n, n))

        for j in range(K):
            if regime_probs_t[j] < 1e-14:
                means_t[j] = np.zeros(n)
                covs_t[j] = np.eye(n) * 1e6
                continue

            weights_i_given_j = pair_weights_post[:, j] / regime_probs_t[j]

            m_j, P_j = collapse_gaussian_mixture(
                means=pair_means[:, j, :],
                covs=pair_covs[:, j, :, :],
                weights=weights_i_given_j,
            )

            means_t[j] = m_j
            covs_t[j] = P_j

        filtered_means[t] = means_t
        filtered_covs[t] = covs_t
        regime_probs[t] = regime_probs_t

        means_prev = means_t
        covs_prev = covs_t
        regime_probs_prev = regime_probs_t

    # Collapse per-regime filtered distributions into a single Gaussian at each t
    state_mean = np.zeros((T, n))
    state_cov = np.zeros((T, n, n))
    for t in range(T):
        state_mean[t], state_cov[t] = collapse_gaussian_mixture(
            filtered_means[t], filtered_covs[t], regime_probs[t]
        )

    return KimFilterResult(
        loglik=total_loglik,
        filtered_means=filtered_means,
        filtered_covs=filtered_covs,
        regime_probs=regime_probs,
        predicted_regime_probs=predicted_regime_probs,
        state_mean=state_mean,
        state_cov=state_cov,
        innovations=innovations,
    )


@dataclass
class SimulateResult:
    y: np.ndarray   # shape (T, n_obs)   — observations
    z: np.ndarray   # shape (T, n_state) — latent states (augmented)
    s: np.ndarray   # shape (T,) int     — regime sequence


def simulate(
    model: RegimeMacroModel,
    T: int,
    seed: Optional[int] = None,
) -> SimulateResult:
    """
    Draw a synthetic path (y, z, s) from the model.

    The augmented state z includes x_{t-1} in position 5.  At t=0 there is no
    lagged observation, so x_{-1} is set to 0 and the initial interest rate
    entering b_t is 0.
    """
    rng = np.random.default_rng(seed)
    p = model.params
    K = model.dims.n_regimes
    n = model.dims.n_state
    m_obs = model.dims.n_obs

    pi0 = p.init_regime_probs if p.init_regime_probs is not None else np.ones(K) / K
    m0 = p.init_state_mean if p.init_state_mean is not None else np.zeros(n)
    P0 = p.init_state_cov if p.init_state_cov is not None else np.eye(n)

    s = np.zeros(T, dtype=int)
    z = np.zeros((T, n))
    y = np.zeros((T, m_obs))

    # Draw initial regime and state
    s[0] = rng.choice(K, p=pi0)
    L0 = np.linalg.cholesky(P0 + 1e-10 * np.eye(n))
    z[0] = m0 + L0 @ rng.standard_normal(n)
    z[0, 5] = 0.0  # x_{-1} unknown; set to zero

    H0 = model.H(s[0], 0, y)
    b0 = model.b(s[0], 0, y)   # uses y[0,3]=0 as initial i_lag
    R0 = model.R(s[0])
    L_R = np.linalg.cholesky(R0)
    y[0] = H0 @ z[0] + b0 + L_R @ rng.standard_normal(m_obs)

    for t in range(1, T):
        P_trans = model.transition_probs(t)
        s[t] = rng.choice(K, p=P_trans[s[t - 1]])

        A_s = model.A(s[t])
        a_s = model.a(s[t])
        sig = model.params.state_sigmas[s[t]]   # shape (5,)

        eps = np.zeros(n)
        eps[:5] = sig * rng.standard_normal(5)
        z[t] = A_s @ z[t - 1] + a_s + eps

        H_s = model.H(s[t], t, y)
        b_s = model.b(s[t], t, y)   # uses y[t-1, 3] which is already set
        R_s = model.R(s[t])
        L_R = np.linalg.cholesky(R_s)
        y[t] = H_s @ z[t] + b_s + L_R @ rng.standard_normal(m_obs)

    return SimulateResult(y=y, z=z, s=s)


@dataclass
class KimSmootherResult:
    smoothed_means:        np.ndarray  # shape (T, K, n_state)
    smoothed_covs:         np.ndarray  # shape (T, K, n_state, n_state)
    smoothed_regime_probs: np.ndarray  # shape (T, K)
    state_mean:            np.ndarray  # shape (T, n_state) — regime-weighted collapsed
    state_cov:             np.ndarray  # shape (T, n_state, n_state)


def kim_smoother(
    model: RegimeMacroModel,
    filter_result: KimFilterResult,
    covariates: Optional[np.ndarray] = None,
) -> KimSmootherResult:
    """
    Kim backward smoother for the regime-switching state-space model.

    Runs a backward pass over the Kim filter output, producing smoothed
    regime probabilities P(s_t | y_{1:T}) and smoothed state distributions
    E[z_t | y_{1:T}, s_t=j] for each regime j.  The collapsed (regime-weighted)
    state mean and covariance are also returned.

    Reference: Kim & Nelson (1999), "State-Space Models with Regime Switching", ch. 5.
    """
    T = filter_result.filtered_means.shape[0]
    K = model.dims.n_regimes
    n = model.dims.n_state

    smoothed_means = filter_result.filtered_means.copy()       # (T, K, n)
    smoothed_covs  = filter_result.filtered_covs.copy()        # (T, K, n, n)
    smoothed_probs = filter_result.regime_probs.copy()         # (T, K)

    filtered_probs   = filter_result.regime_probs              # (T, K)
    predicted_probs  = filter_result.predicted_regime_probs    # (T, K)

    eye_n = np.eye(n)

    for t in range(T - 2, -1, -1):
        P_trans = model.transition_probs(t + 1, covariates)    # (K, K): P(s_{t+1}=k | s_t=j)

        # pair_w[j, k] = P(s_t=j, s_{t+1}=k | y_{1:T})
        pair_w = np.zeros((K, K))
        pair_smooth_means = np.zeros((K, K, n))
        pair_smooth_covs  = np.zeros((K, K, n, n))

        for j in range(K):
            m_filt_j = filter_result.filtered_means[t, j]
            P_filt_j = filter_result.filtered_covs[t, j]

            for k in range(K):
                A_k = model.A(k)
                a_k = model.a(k)
                Q_k = model.Q(k)

                m_pred, P_pred = kalman_predict(m_filt_j, P_filt_j, A_k, a_k, Q_k)

                # Smoother gain: J = P_{t|t}^j  A_k^T  (P_{t+1|t}^{jk})^{-1}
                J = P_filt_j @ A_k.T @ np.linalg.solve(P_pred, eye_n)

                m_s = m_filt_j + J @ (smoothed_means[t + 1, k] - m_pred)
                P_s = P_filt_j + J @ (smoothed_covs[t + 1, k] - P_pred) @ J.T
                P_s = symmetrize(P_s)

                pair_smooth_means[j, k] = m_s
                pair_smooth_covs[j, k]  = P_s

                # P(s_t=j, s_{t+1}=k | y_{1:T})
                pair_w[j, k] = (
                    smoothed_probs[t + 1, k]
                    * filtered_probs[t, j]
                    * P_trans[j, k]
                    / (predicted_probs[t + 1, k] + 1e-300)
                )

        # Smoothed regime probabilities P(s_t=j | y_{1:T}) = sum_k pair_w[j, k]
        new_probs_t = pair_w.sum(axis=1)
        smoothed_probs[t] = new_probs_t

        # Collapse mixture over k for each current regime j
        for j in range(K):
            if new_probs_t[j] < 1e-14:
                smoothed_means[t, j] = np.zeros(n)
                smoothed_covs[t, j]  = np.eye(n) * 1e6
                continue

            cond_w = pair_w[j] / new_probs_t[j]
            smoothed_means[t, j], smoothed_covs[t, j] = collapse_gaussian_mixture(
                pair_smooth_means[j], pair_smooth_covs[j], cond_w
            )

    # Collapse over regime axis to get a single Gaussian at each t
    state_mean = np.zeros((T, n))
    state_cov  = np.zeros((T, n, n))
    for t in range(T):
        state_mean[t], state_cov[t] = collapse_gaussian_mixture(
            smoothed_means[t], smoothed_covs[t], smoothed_probs[t]
        )

    return KimSmootherResult(
        smoothed_means=smoothed_means,
        smoothed_covs=smoothed_covs,
        smoothed_regime_probs=smoothed_probs,
        state_mean=state_mean,
        state_cov=state_cov,
    )


@dataclass
class ForecastResult:
    state_mean:   np.ndarray  # shape (n_ahead, n_state)
    state_cov:    np.ndarray  # shape (n_ahead, n_state, n_state)
    obs_mean:     np.ndarray  # shape (n_ahead, n_obs)
    obs_cov:      np.ndarray  # shape (n_ahead, n_obs, n_obs)
    obs_lower:    np.ndarray  # shape (n_ahead, n_obs) — ci_level lower bound
    obs_upper:    np.ndarray  # shape (n_ahead, n_obs)
    regime_probs: np.ndarray  # shape (n_ahead, K)


def forecast(
    model: RegimeMacroModel,
    filter_result: KimFilterResult,
    y_obs: np.ndarray,
    n_ahead: int,
    ci_level: float = 0.9,
) -> ForecastResult:
    """
    Forecast n_ahead steps beyond the last observation.

    Uses Kim-style Gaussian mixture propagation: at each horizon h, K Gaussian
    components (one per current regime) are propagated forward through all K
    possible next regimes, then collapsed back to K components.

    Confidence intervals are derived from the moments of the resulting Gaussian
    mixture over observations (Gaussian approximation to the marginal).

    The lagged interest rate entering b_t is propagated as the mixture mean of
    the predicted interest rate at each step.
    """
    from scipy.stats import norm as sp_norm

    T = y_obs.shape[0]
    K = model.dims.n_regimes
    n = model.dims.n_state
    m_obs = model.dims.n_obs
    p = model.params

    z_alpha = sp_norm.ppf((1.0 + ci_level) / 2.0)

    # Initialise from last filtered state
    means   = filter_result.filtered_means[-1].copy()   # (K, n)
    covs    = filter_result.filtered_covs[-1].copy()    # (K, n, n)
    weights = filter_result.regime_probs[-1].copy()     # (K,)

    i_lag = y_obs[-1, 3]   # last observed nominal interest rate

    state_mean   = np.zeros((n_ahead, n))
    state_cov    = np.zeros((n_ahead, n, n))
    obs_mean     = np.zeros((n_ahead, m_obs))
    obs_cov      = np.zeros((n_ahead, m_obs, m_obs))
    obs_lower    = np.zeros((n_ahead, m_obs))
    obs_upper    = np.zeros((n_ahead, m_obs))
    regime_probs = np.zeros((n_ahead, K))

    H_mat = model.H(0, 0, None)   # H is constant (regime- and time-independent)

    for h in range(n_ahead):
        P_trans = model.transition_probs(T + h)    # (K, K)

        # Build b for this forecast step using propagated i_lag
        b_h = np.zeros(m_obs)
        b_h[3] = p.rho_i * i_lag - (1.0 - p.rho_i) * p.phi_pi * p.pi_target

        # Propagate K×K pairs and collect predicted state distributions
        pair_means   = np.zeros((K, K, n))
        pair_covs    = np.zeros((K, K, n, n))
        pair_weights = np.zeros((K, K))

        for j in range(K):
            for k in range(K):
                m_pred, P_pred = kalman_predict(
                    means[j], covs[j], model.A(k), model.a(k), model.Q(k)
                )
                pair_means[j, k]   = m_pred
                pair_covs[j, k]    = P_pred
                pair_weights[j, k] = weights[j] * P_trans[j, k]

        new_weights = pair_weights.sum(axis=0)   # P(s_{T+h+1}=k)

        new_means = np.zeros((K, n))
        new_covs  = np.zeros((K, n, n))
        for k in range(K):
            if new_weights[k] < 1e-14:
                new_means[k] = np.zeros(n)
                new_covs[k]  = np.eye(n) * 1e6
                continue
            cond_w = pair_weights[:, k] / new_weights[k]
            new_means[k], new_covs[k] = collapse_gaussian_mixture(
                pair_means[:, k], pair_covs[:, k], cond_w
            )

        # Collapsed state distribution at horizon h+1
        state_mean[h], state_cov[h] = collapse_gaussian_mixture(new_means, new_covs, new_weights)

        # Predicted observation distribution: mixture over regimes
        obs_means_k = np.array([H_mat @ new_means[k] + b_h for k in range(K)])
        obs_covs_k  = np.array([H_mat @ new_covs[k] @ H_mat.T + model.R(k) for k in range(K)])

        obs_mean[h], obs_cov[h] = collapse_gaussian_mixture(obs_means_k, obs_covs_k, new_weights)
        obs_std = np.sqrt(np.maximum(np.diag(obs_cov[h]), 0.0))
        obs_lower[h] = obs_mean[h] - z_alpha * obs_std
        obs_upper[h] = obs_mean[h] + z_alpha * obs_std

        # Propagate i_lag for the next step using the mixture mean interest rate
        i_lag = obs_mean[h, 3]

        regime_probs[h] = new_weights
        means   = new_means
        covs    = new_covs
        weights = new_weights

    return ForecastResult(
        state_mean=state_mean,
        state_cov=state_cov,
        obs_mean=obs_mean,
        obs_cov=obs_cov,
        obs_lower=obs_lower,
        obs_upper=obs_upper,
        regime_probs=regime_probs,
    )


def collapse_gaussian_mixture(means, covs, weights):
    """
    Moment-match a Gaussian mixture into one Gaussian.

    means: shape (M, n)
    covs: shape (M, n, n)
    weights: shape (M,)
    """
    weights = weights / np.sum(weights)

    m = np.sum(weights[:, None] * means, axis=0)

    P = np.zeros_like(covs[0])
    for k in range(len(weights)):
        diff = means[k] - m
        P += weights[k] * (covs[k] + np.outer(diff, diff))

    return m, symmetrize(P)


def logsumexp_2d(X):
    xmax = np.max(X)
    return xmax + np.log(np.sum(np.exp(X - xmax)))

from scipy.optimize import minimize


class ParamPacker:
    """
    Converts between unconstrained optimizer vector theta_raw
    and structured MacroParams.

    This is where you impose:
        sigma > 0
        abs(rho) < 1
        beta_u > 0
        kappa maybe > 0
        transition probabilities via softmax intercepts
    """

    def __init__(self, dims: ModelDims):
        self.dims = dims

    def unpack(self, theta_raw: np.ndarray) -> MacroParams:
        """
        Convert unconstrained vector into MacroParams.

        This function is model-specific. Start simple:
            - shared A
            - regime-specific Q
            - shared R
            - fixed or free transition matrix
        """
        raise NotImplementedError

    def pack(self, params: MacroParams) -> np.ndarray:
        """
        Convert MacroParams into unconstrained vector.
        """
        raise NotImplementedError


def negative_loglik(theta_raw, packer, y, covariates=None):
    try:
        params = packer.unpack(theta_raw)

        dims = packer.dims
        model = RegimeMacroModel(params=params, dims=dims)

        out = kim_filter(model, y, covariates)

        if not np.isfinite(out.loglik):
            return 1e12

        return -out.loglik

    except Exception as e:
        # During optimization, invalid parameter regions are common.
        return 1e12


def fit_mle(
    y: np.ndarray,
    dims: ModelDims,
    theta0_raw: np.ndarray,
    covariates: Optional[np.ndarray] = None,
    method: str = "L-BFGS-B",
):
    packer = ParamPacker(dims)

    result = minimize(
        fun=negative_loglik,
        x0=theta0_raw,
        args=(packer, y, covariates),
        method=method,
        options={
            "maxiter": 1000,
            "disp": True,
        },
    )

    params_hat = packer.unpack(result.x)

    model_hat = RegimeMacroModel(params_hat, dims)
    filter_out = kim_filter(model_hat, y, covariates)

    return {
        "opt_result": result,
        "params_hat": params_hat,
        "filter_out": filter_out,
    }


# Parameter estimation method using RBPF PMMH