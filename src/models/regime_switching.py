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
        self.check_params_validity()

    def __repr__(self):
        return (
            f"RegimeSwitchingSSM("
            f"n_regimes={self.n_regimes}, "
            f"state_dim={self.state_dim}, obs_dim={self.obs_dim})"
        )

    def describe(self):
        return (
            f"{self.__class__.__name__}\n"
            f"  Regime-switching linear Gaussian SSM\n"
            f"  State dim: {self.state_dim},  Obs dim: {self.obs_dim},  Regimes: {self.n_regimes}\n"
            f"  Latent state: (x_t, s_t)  where s_t in {{0, ..., {self.n_regimes - 1}}}\n"
            f"  Transition:  x_t | s_t ~ N(A_{{s_t}} x_{{t-1}},  Q_{{s_t}})\n"
            f"  Observation: y_t | s_t ~ N(C_{{s_t}} x_t,         R_{{s_t}})\n"
            f"  Regime:      P(s_t = j | s_{{t-1}} = i) = P_ij\n"
            f"  Regime transition matrix:\n{self.regime_transition_matrix}\n"
            f"  Stationary regime probs: {self.regime_probabilities_stationary}"
        )

    def check_params_validity(self):
        p = self.regime_transition_matrix
        if np.any(p < 0):
            raise ValueError("regime_transition_matrix has negative entries.")
        if not np.allclose(p.sum(axis=1), 1.0):
            raise ValueError("regime_transition_matrix rows must sum to 1.")
        for i, (Q, R) in enumerate(zip(self.Q_list, self.R_list)):
            if np.any(np.linalg.eigvalsh(Q) < -1e-10):
                raise ValueError(f"Q_list[{i}] is not positive semi-definite.")
            try:
                np.linalg.cholesky(R)
            except np.linalg.LinAlgError:
                raise ValueError(f"R_list[{i}] is not positive definite.")

    def update_params(self, constrained_params):
        # constrained_params: dict with keys matching constructor args
        self.A_list = constrained_params['A_list']
        self.C_list = constrained_params['C_list']
        self.Q_list = constrained_params['Q_list']
        self.R_list = constrained_params['R_list']
        self.regime_transition_matrix = constrained_params['regime_transition_matrix']
        self.regime_probabilities_stationary = self.solve_stationary_distribution()
        self.check_params_validity()

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