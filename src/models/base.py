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
        return f"{self.__class__.__name__}(state_dim={self.state_dim}, obs_dim={self.obs_dim})"

    def describe(self):
        return (
            f"{self.__class__.__name__}\n"
            f"  Latent state dimension: {self.state_dim}\n"
            f"  Observation dimension:  {self.obs_dim}\n"
            f"  Parameters: {self.params_dict}\n"
            f"  Transition:  [not implemented]\n"
            f"  Observation: [not implemented]"
        )

    @property
    def params(self):
        return tuple(self.params_dict.values())

    def check_params_validity(self):
        pass  # override in subclasses to enforce model-specific constraints

    def constrain_params(self, unconstrained_params):
        raise NotImplementedError

    def unconstrain_params(self, constrained_params):
        raise NotImplementedError

    def jacobian_constrain_params(self, unconstrained_params):
        raise NotImplementedError # Needed for MCMC, ideally in closed form

    def constrain_chain(self, ch):
        """Map every row of an unconstrained chain to constrained space.

        Parameters
        ----------
        ch : (n, d) array-like — rows are unconstrained parameter vectors

        Returns
        -------
        (n, d) float array of constrained parameters
        """
        ch = np.asarray(ch, dtype=float)
        d  = len(self.params_dict)

        if ch.ndim == 1:
            raise ValueError(
                f"constrain_chain expects a 2-D array of shape (n, {d}); "
                f"got a 1-D array of length {len(ch)}. "
                "For a single sample use constrain_params."
            )
        if ch.ndim != 2:
            raise ValueError(
                f"constrain_chain expects a 2-D array; got ndim={ch.ndim}."
            )
        if ch.shape[1] != d:
            raise ValueError(
                f"constrain_chain: chain has {ch.shape[1]} columns "
                f"but model has {d} parameters."
            )

        return np.array([self.constrain_params(row) for row in ch], dtype=float)

    def update_params(self, constrained_params):
        # Update all model attributes and params_dict in-place from constrained params.
        # should call check_params_validity() before updating
        raise NotImplementedError

    def clear_state(self):
        # Reset any accumulated mutable runtime state between runs. Default: no-op.
        pass

    def sample_initial_distribution(self):
        # Sample x_0 from the initial distribution.
        # Should use the stationary distribution where possible.
        raise NotImplementedError

    def initial_density(self, x):
        # Return the density p(x_0 = x) of the initial distribution.
        raise NotImplementedError

    def log_initial_density(self, x):
        # Return the log density log p(x_0 = x) of the initial distribution.
        # can be overwritten with closed form expression instead
        return np.log(self.initial_density(x))

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