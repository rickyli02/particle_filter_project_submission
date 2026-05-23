import numpy as np
from scipy.stats import multivariate_normal
from scipy.linalg import solve_discrete_lyapunov
from utils import log_normal_pdf_scalar
from estimation.resampling_methods import systematic_resample
from models.base import StateSpaceModel



# Simple Linear Gaussian state-space model, 1D latent state and observation
# Question: should the observation equation include a constant intercept? y_t = mu + alpha * x_t + nu_t
class SimpleLinearGaussianSSM(StateSpaceModel):
    def __init__(self, phi, alpha, sigma, tau, seed=None):
        super().__init__(seed=seed, state_dim=1, obs_dim=1)
        self.phi = phi
        self.alpha = alpha
        self.sigma = sigma
        self.tau = tau
        self.params_dict = {'phi': phi, 'alpha': alpha, 'sigma': sigma, 'tau': tau}

        self.rng = np.random.default_rng(seed)
    
    def __repr__(self):
        return (
            f"SimpleLinearGaussianSSM("
            f"phi={self.phi!r}, alpha={self.alpha!r}, "
            f"sigma={self.sigma!r}, tau={self.tau!r})"
        )

    def describe(self):
        return (
            f"{self.__class__.__name__}\n"
            f"  Simple linear Gaussian SSM — 1D latent state and observation\n"
            f"  Parameters: {self.params_dict}\n"
            f"  Transition:  x_t = {self.phi} * x_(t-1) + eps_t,   eps_t ~ N(0, {self.sigma}^2)\n"
            f"  Observation: y_t = {self.alpha} * x_t + nu_t,       nu_t  ~ N(0, {self.tau}^2)\n"
            f"  Initial:     x_0 ~ N(0, {self.stationary_var:.6g})  [stationary]"
        )
    
    @property
    def stationary_var(self):
        return self.sigma ** 2 / (1 - self.phi ** 2) if abs(self.phi) < 1 else np.inf
    

    def sample_initial_distribution(self):
        # x_0 ~ N(0, sigma^2 / (1 - phi^2)) for stationarity
        return self.rng.normal(0, np.sqrt(self.stationary_var))

    def initial_density(self, x):
        from scipy.stats import norm
        return norm.pdf(x, loc=0, scale=np.sqrt(self.stationary_var))

    def transition(self, x_prev):
        # x_next = phi * x_prev + eps,   eps ~ N(0, sigma^2)
        # deal with cases where x_next is a scalar and a numpy array
        if np.isscalar(x_prev):
            x_prev = np.array([x_prev])
        return self.phi * x_prev + self.rng.normal(0, self.sigma, size=x_prev.shape)

    def observation(self, x):
        # y_t = alpha * x_t + eta,   eta ~ N(0, tau^2)
        if np.isscalar(x):
            x = np.array([x])
        return self.alpha * x + self.rng.normal(0, self.tau, size=x.shape)

    def log_transition_density(self, x_next, x_prev):
        return log_normal_pdf_scalar(x_next, self.phi * x_prev, self.sigma ** 2)

    def log_observation_density(self, y, x):
        return log_normal_pdf_scalar(y, self.alpha * x, self.tau ** 2)


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