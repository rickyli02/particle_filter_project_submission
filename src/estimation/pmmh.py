import numpy as np

from models.base import StateSpaceModel
from estimation.particle_filter import ParticleFilter


class PMMH:
    """
    Particle Marginal Metropolis-Hastings (PMMH).

    Targets the marginal posterior p(θ | y) by using the particle filter as an
    unbiased estimator of the marginal likelihood p(y | θ).  Proposals are
    Gaussian random walks in the *unconstrained* parameter space; the model's
    constrain_params / unconstrain_params transforms map between the two spaces.

    The particle filter must already have data and a resample_method set.
    The model must implement constrain_params, unconstrain_params, and update_params.

    Parameters
    ----------
    model           : StateSpaceModel
    particle_filter : ParticleFilter
    n_iter          : int
    step_sizes      : array-like of length d, or None  (defaults to 0.1 per dim)
    theta0          : array-like of length d — initial *unconstrained* params;
                      use model.unconstrain_params(...) to obtain this
    log_prior       : callable(theta_unc) -> float, or None  (flat prior)
    seed            : int
    """

    def __init__(
        self,
        model,
        particle_filter,
        n_iter=2000,
        step_sizes=None,
        theta0=None,
        log_prior=None,
        seed=0,
    ):
        if not isinstance(model, StateSpaceModel):
            raise ValueError("model must be a StateSpaceModel instance.")
        if not isinstance(particle_filter, ParticleFilter):
            raise ValueError("particle_filter must be a ParticleFilter instance.")
        if theta0 is None:
            raise ValueError(
                "theta0 (initial unconstrained parameter vector) is required. "
                "Use model.unconstrain_params(...) to obtain it."
            )

        self.model = model
        self.pf = particle_filter
        self.n_iter = n_iter
        self.log_prior = log_prior if log_prior is not None else lambda _: 0.0
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.theta0 = np.asarray(theta0, dtype=float)
        d = len(self.theta0)
        self.step_sizes = (
            np.ones(d) * 0.1
            if step_sizes is None
            else np.asarray(step_sizes, dtype=float)
        )

        if len(self.step_sizes) != d:
            raise ValueError(
                f"step_sizes length {len(self.step_sizes)} != theta0 length {d}."
            )

        self.chain = None
        self.loglik_chain = None
        self.accepted = None
        self.accept_rate = None

    def __repr__(self):
        return (
            f"PMMH(model={self.model!r}, n_iter={self.n_iter}, "
            f"N_particles={self.pf.N_particles})"
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _evaluate_loglik(self, theta_unc):
        """
        Constrain θ, update model via model.update_params, reset model and PF
        state, run the particle filter, and return the log marginal likelihood.

        Returns -inf for proposals that produce an invalid model.
        """
        constrained = self.model.constrain_params(theta_unc)
        self.model.update_params(constrained)
        self.model.clear_state()

        self.pf.particle_history.clear()
        self.pf.weight_history.clear()
        self.pf.resample_history.clear()

        *_, loglik = self.pf.run_filter()
        return loglik if np.isfinite(loglik) else -np.inf

    # ── main sampler ──────────────────────────────────────────────────────────

    def run(self):
        """
        Run PMMH for n_iter iterations.

        Returns
        -------
        chain        : (n_iter + 1, d)  unconstrained samples; row 0 is theta0
        loglik_chain : (n_iter + 1,)    log p̂(y | θ) at each sample
        accepted     : (n_iter,) bool   True where the proposal was accepted
        """
        rng = self.rng
        d = len(self.theta0)

        chain = np.zeros((self.n_iter + 1, d))
        loglik_chain = np.zeros(self.n_iter + 1)
        accepted = np.zeros(self.n_iter, dtype=bool)

        theta_curr = self.theta0.copy()
        loglik_curr = self._evaluate_loglik(theta_curr)
        log_prior_curr = self.log_prior(theta_curr)

        chain[0] = theta_curr
        loglik_chain[0] = loglik_curr

        for i in range(self.n_iter):
            theta_prop = theta_curr + rng.normal(0.0, self.step_sizes)

            try:
                loglik_prop = self._evaluate_loglik(theta_prop)
            except Exception:
                # Invalid proposal (e.g. non-PSD covariance, unstable dynamics)
                chain[i + 1] = theta_curr
                loglik_chain[i + 1] = loglik_curr
                continue

            log_prior_prop = self.log_prior(theta_prop)
            log_alpha = (log_prior_prop + loglik_prop) - (log_prior_curr + loglik_curr)

            if np.log(rng.uniform()) < log_alpha:
                theta_curr = theta_prop
                loglik_curr = loglik_prop
                log_prior_curr = log_prior_prop
                accepted[i] = True

            chain[i + 1] = theta_curr
            loglik_chain[i + 1] = loglik_curr

        self.chain = chain
        self.loglik_chain = loglik_chain
        self.accepted = accepted
        self.accept_rate = float(accepted.mean())

        # Restore model to the last accepted state
        self.model.update_params(self.model.constrain_params(theta_curr))

        return chain, loglik_chain, accepted


class BlockPMMH(PMMH):
    """
    Block-update PMMH.

    Each iteration cycles through a set of parameter blocks, proposing and
    accepting / rejecting each block independently while running the full
    particle filter for each.  This improves mixing when parameters have very
    different scales or are weakly coupled.

    Parameters
    ----------
    blocks : list of list of int, or None
             Index groups, e.g. [[0, 1], [2, 3]].
             None defaults to one block per parameter (component-wise MH).
    """

    def __init__(
        self,
        model,
        particle_filter,
        n_iter=2000,
        step_sizes=None,
        theta0=None,
        log_prior=None,
        seed=0,
        blocks=None,
    ):
        super().__init__(model, particle_filter, n_iter, step_sizes, theta0, log_prior, seed)
        d = len(self.theta0)
        self.blocks = blocks if blocks is not None else [[i] for i in range(d)]

    def __repr__(self):
        return (
            f"BlockPMMH(model={self.model!r}, n_iter={self.n_iter}, "
            f"N_particles={self.pf.N_particles}, n_blocks={len(self.blocks)})"
        )

    def run(self):
        """
        Run block-update PMMH.

        Returns same structure as PMMH.run().  accepted[i] is True if at least
        one block proposal was accepted during iteration i.
        """
        rng = self.rng
        d = len(self.theta0)

        chain = np.zeros((self.n_iter + 1, d))
        loglik_chain = np.zeros(self.n_iter + 1)
        accepted = np.zeros(self.n_iter, dtype=bool)

        theta_curr = self.theta0.copy()
        loglik_curr = self._evaluate_loglik(theta_curr)
        log_prior_curr = self.log_prior(theta_curr)

        chain[0] = theta_curr
        loglik_chain[0] = loglik_curr

        for i in range(self.n_iter):
            for block in self.blocks:
                idx = np.array(block)
                theta_prop = theta_curr.copy()
                theta_prop[idx] += rng.normal(0.0, self.step_sizes[idx])

                try:
                    loglik_prop = self._evaluate_loglik(theta_prop)
                except Exception:
                    continue

                log_prior_prop = self.log_prior(theta_prop)
                log_alpha = (
                    (log_prior_prop + loglik_prop) - (log_prior_curr + loglik_curr)
                )

                if np.log(rng.uniform()) < log_alpha:
                    theta_curr = theta_prop
                    loglik_curr = loglik_prop
                    log_prior_curr = log_prior_prop
                    accepted[i] = True

            chain[i + 1] = theta_curr
            loglik_chain[i + 1] = loglik_curr

        self.chain = chain
        self.loglik_chain = loglik_chain
        self.accepted = accepted
        self.accept_rate = float(accepted.mean())

        self.model.update_params(self.model.constrain_params(theta_curr))

        return chain, loglik_chain, accepted
