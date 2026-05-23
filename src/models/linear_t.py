# similar to linear_gaussian but latent process noise is t-distributed instead of Gaussian
# the observation equation is the same as in linear_gaussian

import numpy as np
from utils import log_normal_pdf_scalar, logsumexp
from estimation.resampling_methods import systematic_resample
from models.base import StateSpaceModel

# import t distribution sampling and pdf from scipy.stats
from scipy.stats import t
from scipy.stats import multivariate_normal

class LinearTSSM(StateSpaceModel):
    def __init__(self, alpha, tau, phi, sigma, df, seed=None):
        super().__init__(seed=seed, state_dim=1, obs_dim=1)
        self.alpha = alpha
        self.tau = tau
        self.phi = phi
        self.sigma = sigma
        self.df = df
        self.params_dict = {'alpha': alpha, 'tau': tau, 'phi': phi, 'sigma': sigma, 'df': df}

        self.rng = np.random.default_rng(seed)

    def __repr__(self):
        return (
            f"LinearTSSM("
            f"phi={self.phi!r}, alpha={self.alpha!r}, sigma={self.sigma!r}, "
            f"tau={self.tau!r}, df={self.df!r})"
        )

    def describe(self):
        return (
            f"{self.__class__.__name__}\n"
            f"  Linear T SSM — 1D latent state and observation\n"
            f"  Parameters: {self.params_dict}\n"
            f"  Transition:  x_t = {self.phi} * x_(t-1) + {self.sigma} * eps_t,   eps_t ~ t(df={self.df})\n"
            f"  Observation: y_t = {self.alpha} * x_t + nu_t,                      nu_t  ~ N(0, {self.tau}^2)"
        )

    def update_params(self, constrained_params):
        alpha, tau, phi, sigma, df = constrained_params
        self.alpha = alpha
        self.tau = tau
        self.phi = phi
        self.sigma = sigma
        self.df = df
        self.params_dict = {'alpha': alpha, 'tau': tau, 'phi': phi, 'sigma': sigma, 'df': df}

    def sample_initial_distribution(self):
        # stationary distribution, x_0 ~ t(df, 0, sigma^2 / (1 - phi^2))
        scale = np.sqrt(self.sigma ** 2 / (1 - self.phi ** 2))
        return self.rng.standard_t(self.df) * scale

    def initial_density(self, x):
        scale = np.sqrt(self.sigma ** 2 / (1 - self.phi ** 2))
        return t.pdf(x, df=self.df, loc=0, scale=scale)
    
    def transition(self, x_prev):
        scale = self.sigma
        return self.phi * x_prev + self.rng.standard_t(self.df) * scale

    def transition_density(self, x_curr, x_prev):
        scale = self.sigma
        # Compute the density of the transition from x_prev to x_curr
        # This is a bit tricky since we're dealing with a t-distributed noise term
        # We'll approximate it using the fact that the difference is t-distributed
        diff = x_curr - self.phi * x_prev
        return t.pdf(diff, df=self.df, loc=0, scale=scale)

    def log_transition_density(self, x_next, x_prev):
        scale = self.sigma
        diff = x_next - self.phi * x_prev
        return t.logpdf(diff, df=self.df, loc=0, scale=scale)

    def observation(self, x):
        return self.alpha * x + self.rng.normal(0, self.tau)

    def log_observation_density(self, y, x):
        return log_normal_pdf_scalar(y, self.alpha * x, self.tau ** 2)

    
    