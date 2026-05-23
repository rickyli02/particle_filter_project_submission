
class ResamplingMethod:
    def __init__(self, seed=None):
        self.resample_threshold = 0.5

        self.seed = seed if seed is not None else 42
        self.rng = np.random.default_rng(seed)

    def resample(self, particles, weights):
        raise NotImplementedError("Resampling method must implement resample(particles, weights)")


# Note that multinomial resampling should not be used in practice due to its high variance, 
# but we include it here for completeness and as a baseline for comparison with other methods.
class MultinomialResampling(ResamplingMethod):
    def resample(self, particles, weights):
        N = len(particles)
        indices = self.rng.choice(N, size=N, p=weights)
        return particles[indices]

class ResidualResampling(ResamplingMethod):
    def resample(self, particles, weights):
        N = len(particles)
        indices = np.zeros(N, dtype=int)

        # Compute the number of deterministic copies for each particle
        num_copies = np.floor(weights * N).astype(int)
        residuals = weights * N - num_copies

        # Fill in the deterministic copies
        idx = 0
        for i in range(N):
            for _ in range(num_copies[i]):
                indices[idx] = i
                idx += 1

        # Resample the remaining particles using multinomial resampling on the residuals
        if idx < N:
            residual_weights = residuals / residuals.sum()
            remaining_indices = self.rng.choice(N, size=N - idx, p=residual_weights)
            indices[idx:] = remaining_indices

        return particles[indices]

class StratifiedResampling(ResamplingMethod):
    def resample(self, particles, weights):
        N = len(particles)
        positions = (self.rng.random(N) + np.arange(N)) / N
        cumsum = np.cumsum(weights)
        indices = np.zeros(N, dtype=int)
        i = j = 0
        while i < N:
            if positions[i] < cumsum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        return particles[indices]

# Systematic resampling should be the default choice
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


