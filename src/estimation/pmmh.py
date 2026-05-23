
# Regular Particle Marginal Metropolis-Hastings (PMMH) given an abstract state-space model and a particle filter implementation.

class PMMH:
    def __init__(self, model, particle_filter, n_iter=2000, step_sizes=None, theta0=None, seed=0):
        self.model = model
        self.particle_filter = particle_filter
        self.n_iter = n_iter
        self.step_sizes = step_sizes if step_sizes is not None else [0.1] * len(theta0)
        self.theta0 = theta0
        self.seed = seed

        # Check theta0, step_sizes, and model parameter dimensions
        if self.theta0 is not None and len(self.theta0) != len(self.step_sizes):
            raise ValueError("Length of theta0 must match length of step_sizes.")
        if self.theta0 is not None and len(self.theta0) != len(self.model.parameters):
            raise ValueError("Length of theta0 must match number of model parameters.") 

    def __repr__(self):
        return f"PMMH(model={self.model}, particle_filter={self.particle_filter}, n_iter={self.n_iter})"

    def proposal_distribution(self, theta):
        # Simple Gaussian random walk proposal
        return np.random.normal(theta, self.step_sizes)
    
    def proposal_log_density(self):
        pass

    def acceptance_ratio(self):
        pass

    def run(self, data):
        # replce the following with correct ''' docstring
        # Implementation of the PMMH algorithm goes here
        # requires model to have contrain / unconstrain methods
        # uses estimated log likelihood from particle filter for acceptance ratio
        pass # returns estimated parameters of the model



# Block-update PMMH
class BlockPMMH(PMMH):
    def __init__(self, model, particle_filter, n_iter=2000, step_sizes=None, theta0=None, seed=0):
        super().__init__(model, particle_filter, n_iter, step_sizes, theta0, seed)

    def run(self, data):
        # Implementation of the block-update PMMH algorithm goes here
        pass