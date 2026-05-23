import numpy as np

# Abstract class for state-space models
class StateSpaceModel:
    def __init__(self, seed=None, state_dim=None, obs_dim=None):
        # include model parameters
        self.params_dict = {}
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        # random seed for reproducibility
        self.seed = seed if seed is not None else 42



    def __repr__(self):
        description = f"""{self.__class__.__name__}
        Latent state dimension: {self.state_dim}
        Observation dimension: {self.obs_dim}
        Parameters: {self.params_dict}
        Transition: transition equation here
        Observation: observation equation here
        """
        return description

    @property
    def params(self):
        return tuple(self.params_dict.values())

    def constrain_params(self, unconstrained_params):
        raise NotImplementedError

    def unconstrain_params(self, constrained_params):
        raise NotImplementedError

    def sample_initial_distribution(self):
        # Sample x_0 from the initial distribution.
        # Should use the stationary distribution where possible.
        raise NotImplementedError

    def initial_density(self, x):
        # Return the density p(x_0 = x) of the initial distribution.
        raise NotImplementedError

    def transition(self, x_prev):
        raise NotImplementedError

    def observation(self, x):
        raise NotImplementedError

    def log_transition_density(self, x_next, x_prev):
        raise NotImplementedError

    def log_observation_density(self, y, x):
        raise NotImplementedError

    def generate_data(self, num_time_steps):
        states = np.zeros((num_time_steps, self.state_dim))
        observations = np.zeros((num_time_steps, self.obs_dim))

        states[0] = self.sample_initial_distribution()
        observations[0] = self.observation(states[0])
        log_likelihood = self.log_observation_density(observations[0], states[0])

        for t in range(1, num_time_steps):
            states[t] = self.transition(states[t-1])
            observations[t] = self.observation(states[t])
            log_likelihood += self.log_observation_density(observations[t], states[t])

        return states, observations, log_likelihood