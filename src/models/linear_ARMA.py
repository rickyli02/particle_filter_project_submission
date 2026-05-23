'''
Latent process:

    x_t = c + phi x_{t-1}
            + nu_t
            + theta_1 nu_{t-1}
            + theta_2 nu_{t-2}
            + theta_3 nu_{t-3},

    nu_t ~ N(0, sigma^2).

Observation equation:

    y_t = alpha x_t + eps_t,
    eps_t ~ N(0, tau^2).

Markov state:

    s_t = [x_t, nu_t, nu_{t-1}, nu_{t-2}]'.

Parameters
----------
y : array-like, shape (T,)
    Observed data.
phi : float
    AR coefficient. Must satisfy |phi| < 1.
alpha : float
    Measurement loading.
c : float
    Latent-state intercept.
theta_1, theta_2, theta_3 : float
    MA coefficients.
sigma : float
    Process noise standard deviation.
tau : float
    Measurement noise standard deviation.
'''

import numpy as np
from utils import logsumexp, log_normal_pdf_scalar
from models.base import StateSpaceModel

class LinearARMASSM(StateSpaceModel):
    def __init__(self, phi, alpha, c, theta_1, theta_2, theta_3, sigma, tau, seed=None):
        state_dim = 4  # [x_t, nu_t, nu_{t-1}, nu_{t-2}]
        obs_dim = 1
        super().__init__(seed=seed, state_dim=state_dim, obs_dim=obs_dim)
        self.phi = phi
        self.alpha = alpha
        self.c = c
        self.theta_1 = theta_1
        self.theta_2 = theta_2
        self.theta_3 = theta_3
        self.sigma = sigma
        self.tau = tau
        self.params_dict = {
            'phi': phi,
            'alpha': alpha,
            'c': c,
            'theta_1': theta_1,
            'theta_2': theta_2,
            'theta_3': theta_3,
            'sigma': sigma,
            'tau': tau
        }

        self.s = None # current state, shape (state_dim,)
        self.s_history = []

        self.seed = seed if seed is not None else 42
        self.rng = np.random.default_rng(seed)  

    def transition(self, s_prev):
        x_prev, nu_prev, nu_prev_1, nu_prev_2 = s_prev
        nu_t = self.rng.normal(0, self.sigma)
        x_t = self.c + self.phi * x_prev + nu_t + self.theta_1 * nu_prev + self.theta_2 * nu_prev_1 + self.theta_3 * nu_prev_2
        return np.array([x_t, nu_t, nu_prev, nu_prev_1])

    def observation(self, s):
        x_t = s[0]
        return self.alpha * x_t + self.rng.normal(0, self.tau)

    def log_transition_density(self, s_next, s_prev):
        '''
        s_prev = [x_{t-1}, nu_{t-1}, nu_{t-2}, nu_{t-3}]
        s_next = [x_t, nu_t, nu_{t-1}, nu_{t-2}]
        p(s_next | s_prev) = p(x_t | s_prev) p(nu_t | s_prev)
        where x_t | s_prev ~ N(c + phi * x_{t-1} + theta_1 * nu_{t-1} + theta_2 * nu_{t-2} + theta_3 * nu_{t-3}, sigma^2)
              nu_t | s_prev ~ N(0, sigma^2)
        '''
        x_prev, nu_prev, nu_prev_1, nu_prev_2 = s_prev
        x_next, nu_next, _, _ = s_next
        mean_x_next = self.c + self.phi * x_prev + self.theta_1 * nu_prev + self.theta_2 * nu_prev_1 + self.theta_3 * nu_prev_2
        log_density_x_next = log_normal_pdf_scalar(x_next, mean_x_next, self.sigma ** 2)
        log_density_nu_next = log_normal_pdf_scalar(nu_next, 0, self.sigma ** 2)
        return log_density_x_next + log_density_nu_next

    def log_observation_density(self, y, s):
        x_t = s[0]
        return log_normal_pdf_scalar(y, self.alpha * x_t, self.tau ** 2)

    def sample_initial_distribution(self):
        # not from the stationary distribution, but should be fine for testing
        x_0 = self.rng.normal(self.c, self.sigma)
        nu_0 = self.rng.normal(0, self.sigma)
        return np.array([x_0, nu_0, nu_0, nu_0])