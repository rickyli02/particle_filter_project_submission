import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.linalg import solve_discrete_lyapunov, cho_factor, cho_solve
from utils import log_normal_pdf_scalar, symmetrize
from estimation.resampling_methods import systematic_resample
from models.base import StateSpaceModel


# General multivariate linear Gaussian state-space model
#
#   Transition:  x_t = A x_{t-1} + b + eps_t,   eps_t ~ N(0, Q)
#   Observation: y_t = C x_t     + d + nu_t,    nu_t  ~ N(0, R)
#   Initial:     x_0 ~ N(mu_0, P_0)
#
# If mu_0 / P_0 are None, the stationary distribution is used (requires A stable).
class LinearGaussianSSM(StateSpaceModel):
    def __init__(self, a, c, q, r, b=None, d=None, mu_0=None, p_0=None, seed=None):
        n = a.shape[0]
        m = c.shape[0]
        super().__init__(seed=seed, state_dim=n, obs_dim=m)

        self.A   = np.atleast_2d(a)
        self.C   = np.atleast_2d(c)
        self.Q   = np.atleast_2d(q)
        self.R   = np.atleast_2d(r)
        self.b   = np.zeros(n) if b is None else np.asarray(b, dtype=float)
        self.d   = np.zeros(m) if d is None else np.asarray(d, dtype=float)
        self.rng = np.random.default_rng(seed)

        # Initial distribution
        if mu_0 is None and p_0 is None:
            self.mu_0, self.P_0 = self._stationary_distribution()
        else:
            self.mu_0 = np.zeros(n) if mu_0 is None else np.asarray(mu_0, dtype=float)
            self.P_0  = np.eye(n)   if p_0 is None else np.atleast_2d(p_0).astype(float)

        self.params_dict = {'A': self.A, 'C': self.C, 'Q': self.Q, 'R': self.R,
                            'b': self.b, 'd': self.d}
        self.check_params_validity()

    def check_params_validity(self):
        n, m = self.state_dim, self.obs_dim
        if self.A.shape != (n, n):
            raise ValueError(f"A shape {self.A.shape} must be ({n}, {n}).")
        if self.C.shape != (m, n):
            raise ValueError(f"C shape {self.C.shape} must be ({m}, {n}).")
        if self.Q.shape != (n, n):
            raise ValueError(f"Q shape {self.Q.shape} must be ({n}, {n}).")
        if self.R.shape != (m, m):
            raise ValueError(f"R shape {self.R.shape} must be ({m}, {m}).")
        if not np.allclose(self.Q, self.Q.T):
            raise ValueError("Q must be symmetric.")
        if not np.allclose(self.R, self.R.T):
            raise ValueError("R must be symmetric.")
        if np.any(np.linalg.eigvalsh(self.Q) < -1e-10):
            raise ValueError("Q must be positive semi-definite.")
        try:
            np.linalg.cholesky(self.R)
        except np.linalg.LinAlgError:
            raise ValueError("R must be positive definite.")

    def __repr__(self):
        return (
            f"LinearGaussianSSM("
            f"state_dim={self.state_dim}, obs_dim={self.obs_dim})"
        )

    def describe(self):
        q_str = repr(np.diag(self.Q)) if np.all(self.Q == np.diag(np.diag(self.Q))) else repr(self.Q)
        r_str = repr(np.diag(self.R)) if np.all(self.R == np.diag(np.diag(self.R))) else repr(self.R)
        return (
            f"{self.__class__.__name__}\n"
            f"  General linear Gaussian SSM\n"
            f"  State dim: {self.state_dim},  Obs dim: {self.obs_dim}\n"
            f"  Transition:  x_t = A x_{{t-1}} + b + eps_t,  eps_t ~ N(0, Q)\n"
            f"  Observation: y_t = C x_t + d + nu_t,          nu_t  ~ N(0, R)\n"
            f"  A:\n{self.A}\n"
            f"  C:\n{self.C}\n"
            f"  Q (diag): {q_str}\n"
            f"  R (diag): {r_str}"
        )

    def update_params(self, constrained_params):
        # constrained_params: dict with keys 'A', 'C', 'Q', 'R', 'b', 'd'
        self.A = np.atleast_2d(constrained_params['A'])
        self.C = np.atleast_2d(constrained_params['C'])
        self.Q = np.atleast_2d(constrained_params['Q'])
        self.R = np.atleast_2d(constrained_params['R'])
        self.b = np.asarray(constrained_params['b'], dtype=float)
        self.d = np.asarray(constrained_params['d'], dtype=float)
        self.params_dict = {'A': self.A, 'C': self.C, 'Q': self.Q, 'R': self.R,
                            'b': self.b, 'd': self.d}
        self.check_params_validity()

    def _stationary_distribution(self):
        """Solve the discrete Lyapunov equation P = A P A' + Q for the stationary covariance."""
        eigenvalues = np.linalg.eigvals(self.A)
        if np.any(np.abs(eigenvalues) >= 1.0):
            raise ValueError(
                "A is not stable (eigenvalue magnitude >= 1). "
                "Provide mu_0 and P_0 explicitly."
            )
        P = solve_discrete_lyapunov(self.A, self.Q)
        mu = np.linalg.solve(np.eye(self.state_dim) - self.A, self.b)
        return mu, P

    def sample_initial_distribution(self):
        return self.rng.multivariate_normal(self.mu_0, self.P_0)

    def initial_density(self, x):
        return multivariate_normal.pdf(x, mean=self.mu_0, cov=self.P_0)

    def transition(self, x_prev):
        x_prev = np.asarray(x_prev, dtype=float)
        return self.A @ x_prev + self.b + self.rng.multivariate_normal(
            np.zeros(self.state_dim), self.Q
        )

    def observation(self, x):
        x = np.asarray(x, dtype=float)
        return self.C @ x + self.d + self.rng.multivariate_normal(
            np.zeros(self.obs_dim), self.R
        )

    def log_transition_density(self, x_next, x_prev):
        mean = self.A @ np.asarray(x_prev, dtype=float) + self.b
        return multivariate_normal.logpdf(x_next, mean=mean, cov=self.Q)

    def log_observation_density(self, y, x):
        mean = self.C @ np.asarray(x, dtype=float) + self.d
        return multivariate_normal.logpdf(y, mean=mean, cov=self.R)

    def log_likelihood(self, y):
        """Exact log p(y_{0:T-1} | theta) via multivariate Kalman filter recursion."""
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        T, m = y.shape

        A, C, Q, R, b, d = self.A, self.C, self.Q, self.R, self.b, self.d
        n = self.state_dim
        eye_n = np.eye(n)

        mu = self.mu_0.copy()
        P  = self.P_0.copy()
        loglik = 0.0

        for t in range(T):
            v  = y[t] - C @ mu - d
            S  = symmetrize(C @ P @ C.T + R)

            cf        = cho_factor(S)
            log_det_s = 2.0 * np.sum(np.log(np.diag(cf[0])))
            loglik   -= 0.5 * (m * np.log(2.0 * np.pi) + log_det_s + v @ cho_solve(cf, v))

            K         = cho_solve(cf, C @ P).T          # (n × m)
            i_minus_kc = eye_n - K @ C
            mu = mu + K @ v
            P  = symmetrize(i_minus_kc @ P @ i_minus_kc.T + K @ R @ K.T)

            if t < T - 1:
                mu = A @ mu + b
                P  = symmetrize(A @ P @ A.T + Q)

        return loglik


# Simple 1D linear Gaussian SSM, implemented as a LinearGaussianSSM subclass.
#
#   Transition:  x_t = phi * x_{t-1} + eps_t,   eps_t ~ N(0, sigma2)
#   Observation: y_t = alpha * x_t + nu_t,       nu_t  ~ N(0, tau2)
#   Initial:     x_0 ~ N(0, sigma2 / (1 - phi^2))  [stationary]
#
# Parameterized by theta = (phi, alpha, sigma2, tau2).
# Scalar properties and the (unconstrain/constrain/update)_params interface are
# kept for backward compatibility with existing callers and estimators.
# Because it is a LinearGaussianSSM subclass, KalmanFilter accepts it directly.
#
# log_likelihood and score use the fast scalar Kalman recursion; transition and
# observation keep batched-particle support for use with the particle filter.
class SimpleLinearGaussianSSM(LinearGaussianSSM):
    def __init__(self, phi, alpha, sigma2, tau2, initial_var = None, seed=None):
        # Validate before super() so the error messages are readable (super() calls
        # _stationary_distribution which raises a less specific message for phi >= 1).
        if abs(phi) >= 1:
            raise ValueError(f"phi={phi}: latent process is not stationary (require |phi| < 1).")
        if sigma2 <= 0:
            raise ValueError(f"sigma2={sigma2}: process noise variance must be positive.")
        if tau2 <= 0:
            raise ValueError(f"tau2={tau2}: observation noise variance must be positive.")
        super().__init__(
            a=np.array([[phi]]),
            c=np.array([[alpha]]),
            q=np.array([[sigma2]]),
            r=np.array([[tau2]]),
            seed=seed,
        )
        self._initial_var_fixed = initial_var is not None
        if initial_var is not None:
            self.initial_var = initial_var
            self.P_0 = np.array([[initial_var]])
        else:
            self.initial_var = self.stationary_var
        
        # Override params_dict to expose scalar names instead of matrix names.
        self.params_dict = {'phi': phi, 'alpha': alpha, 'sigma2': sigma2, 'tau2': tau2}

    # ── scalar properties (backward compat) ───────────────────────────────────

    @property
    def phi(self):    return float(self.A[0, 0])

    @property
    def alpha(self):  return float(self.C[0, 0])

    @property
    def sigma2(self): return float(self.Q[0, 0])

    @property
    def tau2(self):   return float(self.R[0, 0])

    @property
    def stationary_var(self):
        return self.sigma2 / (1 - self.phi ** 2) if abs(self.phi) < 1 else np.inf


    # ── validity ──────────────────────────────────────────────────────────────

    def check_params_validity(self):
        super().check_params_validity()   # shapes, Q PSD, R PD
        if abs(self.phi) >= 1:
            raise ValueError(
                f"phi={self.phi}: latent process is not stationary (require |phi| < 1)."
            )

    # ── string representation ─────────────────────────────────────────────────

    def __repr__(self):
        return (
            f"SimpleLinearGaussianSSM("
            f"phi={self.phi!r}, alpha={self.alpha!r}, "
            f"sigma2={self.sigma2!r}, tau2={self.tau2!r})"
        )

    def describe(self):
        return (
            f"{self.__class__.__name__}\n"
            f"  Simple linear Gaussian SSM — 1D latent state and observation\n"
            f"  Parameters: {self.params_dict}\n"
            f"  Transition:  x_t = {self.phi} * x_(t-1) + eps_t,   eps_t ~ N(0, {self.sigma2})\n"
            f"  Observation: y_t = {self.alpha} * x_t + nu_t,       nu_t  ~ N(0, {self.tau2})\n"
            f"  Initial:     x_0 ~ N(0, {self.initial_var:.6g})  [{'fixed' if self._initial_var_fixed else 'stationary'}]"
        )

    # ── parameter interface ───────────────────────────────────────────────────

    def update_params(self, constrained_params):
        phi, alpha, sigma2, tau2 = constrained_params
        # Update the underlying matrices via the parent (also calls check_params_validity).
        super().update_params({
            'A': np.array([[phi]]),
            'C': np.array([[alpha]]),
            'Q': np.array([[sigma2]]),
            'R': np.array([[tau2]]),
            'b': np.zeros(1),
            'd': np.zeros(1),
        })
        # Refresh the stationary initial distribution (phi and sigma2 may have changed).
        # LinearGaussianSSM.update_params does not do this automatically.
        self.mu_0, self.P_0 = self._stationary_distribution()
        if self._initial_var_fixed:
            self.P_0 = np.array([[self.initial_var]])
        else:
            self.initial_var = self.stationary_var
        # Restore the scalar params_dict (parent overwrites it with matrix keys).
        self.params_dict = {'phi': phi, 'alpha': alpha, 'sigma2': sigma2, 'tau2': tau2}

    def unconstrain_params(self, constrained_params):
        # Note: unconstraining near boundaries compresses the posterior — MCMC
        # samplers work better in the unconstrained space.
        phi, alpha, sigma2, tau2 = constrained_params
        return np.array([np.arctanh(phi), alpha, np.log(sigma2), np.log(tau2)])

    def constrain_params(self, unconstrained_params):
        u_phi, u_alpha, u_sigma2, u_tau2 = unconstrained_params
        return [float(np.tanh(u_phi)), float(u_alpha),
                float(np.exp(u_sigma2)), float(np.exp(u_tau2))]

    def jacobian_constrain_params(self, unconstrained_params):
        """
        Diagonal Jacobian of constrain_params at unconstrained_params.

        Returns a (4, 4) diagonal matrix J where J[k, k] = d(θ_con[k]) / d(θ_unc[k]):
          phi   : d(tanh(u)) / du    = 1 - tanh²(u) = 1 - phi²
          alpha : d(u) / du          = 1
          sigma2: d(exp(u)) / du     = exp(u) = sigma²
          tau2  : d(exp(u)) / du     = exp(u) = tau²
        """
        u_phi, _, u_sigma2, u_tau2 = unconstrained_params
        return np.diag([
            1.0 - np.tanh(u_phi) ** 2,
            1.0,
            np.exp(u_sigma2),
            np.exp(u_tau2),
        ])

    # ── sampling and densities ────────────────────────────────────────────────
    # Scalar / batched-particle overrides (parent uses multivariate_normal which
    # does not broadcast over N particles).

    def sample_initial_distribution(self):
        return np.array([self.rng.normal(0, np.sqrt(self.initial_var))])

    def initial_density(self, x):
        return norm.pdf(x, loc=0, scale=np.sqrt(self.initial_var))

    def transition(self, x_prev):
        if np.isscalar(x_prev):
            x_prev = np.array([x_prev])
        return self.phi * x_prev + self.rng.normal(0, np.sqrt(self.sigma2), size=x_prev.shape)

    def observation(self, x):
        if np.isscalar(x):
            x = np.array([x])
        return self.alpha * x + self.rng.normal(0, np.sqrt(self.tau2), size=x.shape)

    def log_transition_density(self, x_next, x_prev):
        return log_normal_pdf_scalar(x_next, self.phi * x_prev, self.sigma2)

    def log_observation_density(self, y, x):
        return log_normal_pdf_scalar(y, self.alpha * x, self.tau2)

    # ── log-likelihood and score ──────────────────────────────────────────────
    # Fast scalar Kalman recursion — O(T) without any matrix operations.

    def log_likelihood(self, y):
        """
        Exact log p(y_{0:T-1} | theta) via scalar Kalman filter recursion.

        y_t | y_{0:t-1} ~ N(alpha * mu_{t|t-1},  alpha^2 * P_{t|t-1} + tau^2)
        Initial:  mu_0 = 0,  P_0 = initial_var (when not provided in initialization, sigma^2 / (1 - phi^2)  [stationary])
        """
        y = np.asarray(y, dtype=float).ravel()
        T = len(y)

        mu = 0.0
        P  = self.initial_var

        loglik = 0.0
        for t in range(T):
            S = self.alpha ** 2 * P + self.tau2
            v = y[t] - self.alpha * mu

            loglik -= 0.5 * (np.log(2.0 * np.pi * S) + v ** 2 / S)

            K  = self.alpha * P / S
            mu = mu + K * v
            P  = (1.0 - K * self.alpha) ** 2 * P + K ** 2 * self.tau2

            if t < T - 1:
                mu = self.phi * mu
                P  = self.phi ** 2 * P + self.sigma2

        return loglik

    def score(self, y):
        """
        Gradient of log p(y_{0:T-1} | theta) w.r.t. theta = (phi, alpha, sigma2, tau2).

        Propagates analytic sensitivities of (mu, P) through the Kalman recursion in one
        forward pass.  Cost is O(T), same order as log_likelihood itself.

        Returns shape-(4,) array  [d/dphi, d/dalpha, d/dsigma2, d/dtau2].
        """
        y = np.asarray(y, dtype=float).ravel()
        T = len(y)
        phi, alpha, sigma2, tau2 = self.phi, self.alpha, self.sigma2, self.tau2
        initial_var = self.initial_var
        phi2 = phi ** 2
        one_minus_phi2 = 1.0 - phi2

        mu = 0.0
        P  = initial_var

        # Sensitivities of (mu, P) w.r.t. (phi, alpha, sigma2, tau2), indices 0-3
        dmu = np.zeros(4)
        if self._initial_var_fixed:
            dp = np.zeros(4)
        else:
            dp = np.array([
                2.0 * phi * sigma2 / one_minus_phi2 ** 2,  # dP0/dphi
                0.0,                                         # dP0/dalpha
                1.0 / one_minus_phi2,                        # dP0/dsigma2
                0.0,                                         # dP0/dtau2
            ])

        grad = np.zeros(4)

        for t in range(T):
            s     = alpha ** 2 * P + tau2
            v     = y[t] - alpha * mu
            inv_s = 1.0 / s

            # dS/dtheta
            ds    = alpha ** 2 * dp
            ds[1] += 2.0 * alpha * P
            ds[3] += 1.0

            # dv/dtheta
            dv    = -alpha * dmu
            dv[1] -= mu

            # Score contribution: ell_t = -0.5*(log(2pi*S) + v^2/S)
            r     = v * inv_s
            grad += -0.5 * (ds * inv_s * (1.0 - v * r) + 2.0 * r * dv)

            # Kalman gain and its sensitivity
            k  = alpha * P * inv_s
            dk = (alpha * dp - k * ds) * inv_s
            dk[1] += P * inv_s

            # Updated mean and its sensitivity
            mu_up  = mu + k * v
            dmu_up = dmu + dk * v + k * dv

            # Updated variance (Joseph form): P_{t|t} = c^2*P + k^2*tau2
            c      = 1.0 - k * alpha
            dc     = -dk * alpha
            dc[1] -= k
            p_up   = c ** 2 * P + k ** 2 * tau2
            dp_up  = 2.0 * c * dc * P + c ** 2 * dp + 2.0 * k * dk * tau2
            dp_up[3] += k ** 2

            if t < T - 1:
                dmu    = phi * dmu_up
                dmu[0] += mu_up

                dp    = phi2 * dp_up
                dp[0] += 2.0 * phi * p_up
                dp[2] += 1.0

                mu = phi * mu_up
                P  = phi2 * p_up + sigma2

        return grad

    def hessian_log_likelihood(self, y):
        raise NotImplementedError


class FixedAlphaSSM(SimpleLinearGaussianSSM):
    """SimpleLinearGaussianSSM with alpha fixed at 1; estimates (phi, sigma2, tau2)."""

    def __init__(self, alpha_fixed, phi, sigma2, tau2, initial_var = None, seed=None):
        self.ALPHA_FIXED = alpha_fixed
        super().__init__(phi=phi, alpha=self.ALPHA_FIXED, sigma2=sigma2, tau2=tau2, initial_var=initial_var, seed=seed)
        self.params_dict = {'phi': phi, 'sigma2': sigma2, 'tau2': tau2}

    def update_params(self, constrained_params):
        phi, sigma2, tau2 = constrained_params
        SimpleLinearGaussianSSM.update_params(self, [phi, self.ALPHA_FIXED, sigma2, tau2])
        self.params_dict = {'phi': phi, 'sigma2': sigma2, 'tau2': tau2}

    def unconstrain_params(self, constrained_params):
        phi, sigma2, tau2 = constrained_params
        return np.array([np.arctanh(phi), np.log(sigma2), np.log(tau2)])

    def constrain_params(self, unconstrained_params):
        u_phi, u_sigma2, u_tau2 = unconstrained_params
        return [float(np.tanh(u_phi)), float(np.exp(u_sigma2)), float(np.exp(u_tau2))]

    def jacobian_constrain_params(self, unconstrained_params):
        u_phi, u_sigma2, u_tau2 = unconstrained_params
        return np.diag([
            1.0 - np.tanh(u_phi) ** 2,
            np.exp(u_sigma2),
            np.exp(u_tau2),
        ])

    def score(self, y):
        """Score w.r.t. free params (phi, sigma2, tau2) — drops the fixed-alpha component."""
        full = SimpleLinearGaussianSSM.score(self, y)   # [d/dphi, d/dalpha, d/dsigma2, d/dtau2]
        return np.array([full[0], full[2], full[3]])