from .base import StateSpaceModel
import numpy as np
from scipy.stats import multivariate_normal


class RegimeSwitchingSSM(StateSpaceModel):
    def __init__(self, A_list, C_list, Q_list, R_list, regime_transition_matrix, regime_probabilities, seed=None):
        state_dim = A_list[0].shape[0]
        obs_dim = C_list[0].shape[0]
        super().__init__(seed=seed, state_dim=state_dim, obs_dim=obs_dim)
        self.A_list = A_list
        self.C_list = C_list
        self.Q_list = Q_list
        self.R_list = R_list
        self.regime_transition_matrix = regime_transition_matrix
        self.regime_probabilities = regime_probabilities
        self.rng = np.random.default_rng(seed)

    def transition(self, x_prev, regime):
        A = self.A_list[regime]
        Q = self.Q_list[regime]
        return A @ x_prev + self.rng.multivariate_normal(np.zeros(A.shape[0]), Q)

    def observation(self, x, regime):
        C = self.C_list[regime]
        R = self.R_list[regime]
        return C @ x + self.rng.multivariate_normal(np.zeros(C.shape[0]), R)

    def log_transition_density(self, x_next, x_prev, regime):
        A = self.A_list[regime]
        Q = self.Q_list[regime]
        return multivariate_normal.logpdf(x_next, mean=A @ x_prev, cov=Q)

    def log_observation_density(self, y, x, regime):
        C = self.C_list[regime]
        R = self.R_list[regime]
        return multivariate_normal.logpdf(y, mean=C @ x, cov=R)