from models.base import StateSpaceModel
import numpy as np
from utils import logsumexp


class ResamplingMethod:
    def __init__(self, seed=None):
        self.resample_threshold = 0.5

        self.seed = seed if seed is not None else 42
        self.rng = np.random.default_rng(seed)

    def resample(self, particles, weights):
        raise NotImplementedError("Resampling method must implement resample(particles, weights)")


class MultinomialResampling(ResamplingMethod):
    def resample(self, particles, weights):
        N = len(particles)
        indices = self.rng.choice(N, size=N, p=weights)
        return particles[indices]


class SystematicResampling(ResamplingMethod):
    def _get_indices(self, weights):
        """Return resampled indices using the systematic scheme."""
        N = len(weights)
        positions = (self.rng.random() + np.arange(N)) / N
        cumsum = np.cumsum(weights)
        indices = np.zeros(N, dtype=int)
        i = j = 0
        while i < N:
            if positions[i] < cumsum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        return indices

    def resample(self, particles, weights):
        return particles[self._get_indices(weights)]


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

        self.particle_history = []
        self.weight_history = []
        self.resample_history = []  # 0/1 for no resample / resample at each step

    @property
    def ESS(self):
        return 1.0 / np.sum(self.weights ** 2) if hasattr(self, 'weights') else None

    def run_filter(self):
        T = self.total_time_steps
        N = self.N_particles
        loglik = 0.0

        self.particles = np.array([self.model.sample_initial_distribution() for _ in range(N)])
        self.weights = np.ones(N) / N

        for t in range(T):
            # Propagate particles through transition model
            self.particles = np.array([self.model.transition(p) for p in self.particles])

            # Compute log-weights and accumulate marginal log-likelihood
            y_t = self.data[t]
            log_weights = np.array([self.model.log_observation_density(y_t, p) for p in self.particles])
            loglik += logsumexp(log_weights) - np.log(N)

            # Normalize weights
            log_weights -= log_weights.max()
            self.weights = np.exp(log_weights)
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

        self.latent_state_estimate = np.average(self.particles, axis=0, weights=self.weights)

        return self.latent_state_estimate, self.particle_history, self.weight_history, self.resample_history, loglik