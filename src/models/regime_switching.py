from .base import StateSpaceModel
import numpy as np
from scipy.stats import multivariate_normal


class RegimeSwitchingSSM(StateSpaceModel):
    def __init__(self, A_list, C_list, Q_list, R_list, regime_transition_matrix, seed=None):
        state_dim = A_list[0].shape[0]
        obs_dim = C_list[0].shape[0]
        super().__init__(seed=seed, state_dim=state_dim, obs_dim=obs_dim)
        self.A_list = A_list
        self.C_list = C_list
        self.Q_list = Q_list
        self.R_list = R_list
        self.n_regimes = len(A_list)
        self.regime_transition_matrix = regime_transition_matrix

        pi = self.solve_stationary_distribution()
        self.regime_probabilities_stationary = pi[:-1]
        
        self.rng = np.random.default_rng(seed)

    def __repr__(self):
        description = f"""{self.__class__.__name__}
        Regime-switching linear Gaussian state-space model.
        State dimension: {self.state_dim}, Observation dimension: {self.obs_dim}
        Latent state: (x_t, s_t) where s_t is the discrete regime at time t
        Number of regimes: {self.n_regimes}
        Transition matrices: {self.A_list}
        Observation matrices: {self.C_list}
        Process noise covariances: {self.Q_list}
        Observation noise covariances: {self.R_list}
        Regime transition matrix: {self.regime_transition_matrix}
        Initial regime probabilities: {self.regime_probabilities_stationary}
        """
        return description

    def solve_stationary_distribution(self):
        # Solve for the stationary distribution of the regime Markov chain

        # Initial regime probabilities should be calculated from the stationary distribution of the regime Markov chain
        # Note: this assumes the regime transition matrix is ergodic and has a unique stationary distribution
        # Solve (I - P)^T * pi^T = 0 with the constraint sum(pi) = 1

        I = np.eye(self.n_regimes)
        A = I - self.regime_transition_matrix.T
        # Add a last row of ones for the sum constraint
        A = np.vstack([A, np.ones(self.n_regimes)])
        b = np.zeros(self.n_regimes + 1)
        b[-1] = 1
        pi = np.linalg.solve(A, b)
        return pi[:-1]
    
    def sample_initial_distribution(self):
        # Sample x_0 from the initial distribution.
        # Sample initial regime from stationary distribution of the regime Markov chain
        initial_regime = self.rng.choice(self.n_regimes, p=self.regime_probabilities_stationary)
        A = self.A_list[initial_regime]
        Q = self.Q_list[initial_regime]
        stationary_cov = np.linalg.solve(np.eye(A.shape[0]) - A @ A.T, Q)
        return self.rng.multivariate_normal(np.zeros(A.shape[0]), stationary_cov), initial_regime

    def initial_density(self, x, regime):
        # Caculate initial density of regime
        prob = self.regime_probabilities_stationary[regime]
        # Calculate the initial density of x given the regime
        A = self.A_list[regime]
        Q = self.Q_list[regime]
        stationary_cov = np.linalg.solve(np.eye(A.shape[0]) - A @ A.T, Q)
        return prob * multivariate_normal.pdf(x, mean=np.zeros(A.shape[0]), cov=stationary_cov)

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