# Should be used for suitable models only
# utilizes regimes and Kalman filter for estimation with lower variance than naive particle filter

class KimFilter:
    def __init__(self, model, n_particles=1000, resampling_method=None, seed=0):
        # Check that model is compatible with Kim filter (e.g. has discrete regimes and linear Gaussian structure)
        if not hasattr(model, 'regimes') or not hasattr(model, 'linear_gaussian'):
            raise ValueError("Model must have 'regimes' and 'linear_gaussian' attributes for Kim filter.")
            
        self.model = model
        self.n_particles = n_particles
        self.resampling_method = resampling_method if resampling_method is not None else SystematicResampling(seed=seed)
        self.seed = seed

    def __repr__(self):
        return f"KimFilter(model={self.model}, n_particles={self.n_particles}, resampling_method={self.resampling_method})"

    def run_filter(self, data):
        # Implementation of the Kim filter algorithm goes here
        pass

    def run_smoother(self, data):
        # Implementation of the Kim smoother algorithm goes here
        # forward-backward smoother
        pass