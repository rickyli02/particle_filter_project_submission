from models.base import StateSpaceModel
import numpy as np
import time
from utils import logsumexp
from estimation.resampling_methods import ResamplingMethod

class ParticleFilter:
    def __init__(self, model=None, N_particles=10000, data=None, resample_method=None, seed=None):
        if model is None:
            raise ValueError("ParticleFilter requires a state space model instance.")
        if data is None:
            raise ValueError("ParticleFilter requires observed data to perform filtering.")
        if resample_method is not None and not isinstance(resample_method, ResamplingMethod):
            raise ValueError("resample_method must be an instance of ResamplingMethod or None.")

        self.model = model
        self.data = data
        self.total_time_steps = len(data)
        self.resample_method = resample_method
        self.resample_threshold = resample_method.resample_threshold if resample_method is not None else 0.5
        self.N_particles = N_particles
        self.seed = seed if seed is not None else 42
        self.rng = np.random.default_rng(seed)

        self.particles = None # current particles at each time step, shape (N_particles, state_dim)
        self.weights = None # current normalized weights, shape (N_particles,)

        self.particle_history = []
        self.weight_history = []
        self.resample_history = []  # 0/1 for no resample / resample at each step

    @property
    def ESS(self):
        return 1.0 / np.sum(self.weights ** 2) if hasattr(self, 'weights') else None

    def run_filter(self, verbose=False):
        '''
        returns
            latent state trajectory estimate
            history of particles and weights
            resampling history (=1 if resampled)
            estimated log likelihood
        '''
        _t0 = time.perf_counter()
        T = self.total_time_steps
        N = self.N_particles
        loglik = 0.0

        # Reset history
        self.particle_history = []
        self.weight_history = []
        self.resample_history = []

        # Initialize x_0
        self.particles = np.array([self.model.sample_initial_distribution() for _ in range(N)])
        self.weights = np.ones(N) / N

        # Weight y_0 using x_0

        for t in range(T):
            # Propagate particles through transition model x_t ~ p(x_t | x_{t-1})
            self.particles = np.array([self.model.transition(p) for p in self.particles])

            # Compute log-weights and accumulate marginal log-likelihood
            y_t = self.data[t]

            # Compute g(y_t | x_t^{i}) for y_t ~ g(y_t | x_t)
            log_g = np.array([
                self.model.log_observation_density(y_t, p)
                for p in self.particles
            ]).flatten()

            # include previous weights
            log_prev_w = np.log(self.weights + 1e-300)
            log_unnorm_w = log_prev_w + log_g

            # Marginal likelihood increment
            # log hat p(y_t | y_{0:t-1}) = log sum_i exp(log W_{t-1}^{i} + log g(y_t | x_t^{i})) = log sum_i W_{t-1}^{i} g(y_t | x_t^{i})
            log_inc = logsumexp(log_unnorm_w)
            loglik += log_inc

            # Normalize weights
            log_norm_w = log_unnorm_w - log_inc
            self.weights = np.exp(log_norm_w)
            self.weights /= self.weights.sum()

            # Store history (before resampling — standard SMC convention)
            self.particle_history.append(self.particles.copy())
            self.weight_history.append(self.weights.copy())

            # Resample if ESS is below threshold
            if self.ESS < self.resample_threshold * N and self.resample_method is not None:
                self.particles = self.resample_method.resample(self.particles, self.weights)
                self.weights.fill(1.0 / N)
                self.resample_history.append(1)
            else:
                self.resample_history.append(0)

        self.latent_state_estimate = self.filtered_trajectory()

        if verbose:
            print(f"ParticleFilter.run_filter  {time.perf_counter() - _t0:.3f}s")
        return self.latent_state_estimate, self.particle_history, self.weight_history, self.resample_history, float(loglik)

    def run_smoother(self):
        # Implementation of the particle smoother algorithm goes here
        # possibilities: forward-backward smoother, two-filter smoother, fixed-lag smoother
        pass

    # Backwards compatability. Use latent_state_estmimate after run_filter()
    def filtered_trajectory(self, state_idx=0):
        # Extract, flatten weights, and stack into a matrix (n_time_steps, n_particles)
        W = np.stack([w.flatten() for w in self.weight_history])
        
        # Extract states, handle shape, and stack into a (n_time_steps, n_particles) matrix
        X = np.stack([p if p.ndim == 1 else p[:, state_idx] for p in self.particle_history])
        
        # Weighted average: Multiply row-wise, sum over the particles axis, and divide by sum of weights
        return np.einsum('ij, ij -> i', X, W) / np.sum(W, axis=1)