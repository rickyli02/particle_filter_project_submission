import numpy as np
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
        description = f"""{self.__class__.__name__}
        Simple linear Gaussian state-space model with 1D latent state and observation.
        Parameters: {self.params_dict}
        Transition: x_t = phi * x_(t-1) + eps_t,   eps_t ~ N(0, sigma^2)
        Observation: y_t = alpha * x_t + nu_t,       nu_t  ~ N(0, tau^2)
        """

        return description
    
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

