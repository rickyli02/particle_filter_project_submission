import numpy as np

from estimation.mcmc import MCMCBase
from estimation.particle_filter import ParticleFilter
from models.base import StateSpaceModel
from utils import timer


class PMMH(MCMCBase):
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
    log_prior       : callable, or None
                      Log prior on constrained θ or unconstrained z; see prior_space.
    prior_space     : {"constrained", "unconstrained"}
                      Whether log_prior is defined on θ or z.
    include_jacobian : bool | None
                      Whether to include log|det dθ/dz| in the target density.
                      Default: True for constrained priors, False for unconstrained priors.
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
        prior_space="constrained",
        include_jacobian=None,
        seed=0,
    ):
        if not isinstance(particle_filter, ParticleFilter):
            raise ValueError("particle_filter must be a ParticleFilter instance.")
        super().__init__(
            model,
            n_iter,
            step_sizes,
            theta0,
            log_prior,
            prior_space,
            include_jacobian,
            seed,
        )
        self.pf = particle_filter

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

    @timer
    def run(self, log_interval=500):
        """
        Run PMMH for n_iter iterations.

        Returns
        -------
        chain        : (n_iter + 1, d)  unconstrained samples; row 0 is theta0
        loglik_chain : (n_iter + 1,)    log p̂(y | θ) at each sample
        accepted     : (n_iter,) bool   True where the proposal was accepted
        """
        rng = self.rng
        d   = len(self.theta0)

        chain        = np.zeros((self.n_iter + 1, d))
        loglik_chain = np.zeros(self.n_iter + 1)
        accepted     = np.zeros(self.n_iter, dtype=bool)

        theta_curr  = self.theta0.copy()
        loglik_curr = self._evaluate_loglik(theta_curr)
        lp_curr     = self._log_prior_and_jac(theta_curr)

        chain[0]        = theta_curr
        loglik_chain[0] = loglik_curr

        for i in range(self.n_iter):
            theta_prop = theta_curr + rng.normal(0.0, self.step_sizes)

            try:
                loglik_prop = self._evaluate_loglik(theta_prop)
            except Exception:
                # Invalid proposal (e.g. non-PSD covariance, unstable dynamics)
                chain[i + 1]        = theta_curr
                loglik_chain[i + 1] = loglik_curr
                continue

            lp_prop   = self._log_prior_and_jac(theta_prop)
            log_alpha = (lp_prop + loglik_prop) - (lp_curr + loglik_curr)

            if np.log(rng.uniform()) < log_alpha:
                theta_curr  = theta_prop
                loglik_curr = loglik_prop
                lp_curr     = lp_prop
                accepted[i] = True

            chain[i + 1]        = theta_curr
            loglik_chain[i + 1] = loglik_curr

            if i > 0 and i % log_interval == 0:
                print(
                    f"[{i}/{self.n_iter}]  theta = {chain.mean(axis=0)},  "
                    f"loglik = {loglik_curr:.2f},  accept rate = {accepted[:i].mean():.3f}"
                )

        self.chain        = chain
        self.loglik_chain = loglik_chain
        self.accepted     = accepted
        self.accept_rate  = float(accepted.mean())

        self.model.update_params(self.model.constrain_params(theta_curr))
        return chain, loglik_chain, accepted


class BlockPMMH(PMMH):
    """
    Block-update PMMH.

    Each iteration cycles through a set of parameter blocks, proposing and
    accepting / rejecting each block independently while running the full
    particle filter for each.  This improves mixing when parameters have very
    different scales or are weakly coupled.

    Fixing parameters: to hold a subset of parameters constant, define a
    subclassed model whose params_dict omits those parameters and whose
    constrain/unconstrain/update_params methods operate only on the free
    parameters (see FixedAlphaSSM in the notebooks for an example).  The
    blocks then index into the free-parameter vector only.

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
        prior_space="constrained",
        include_jacobian=None,
        seed=0,
        blocks=None,
    ):
        super().__init__(
            model,
            particle_filter,
            n_iter,
            step_sizes,
            theta0,
            log_prior,
            prior_space,
            include_jacobian,
            seed,
        )
        d = len(self.theta0)
        self.blocks = blocks if blocks is not None else [[i] for i in range(d)]

    def __repr__(self):
        return (
            f"BlockPMMH(model={self.model!r}, n_iter={self.n_iter}, "
            f"N_particles={self.pf.N_particles}, n_blocks={len(self.blocks)})"
        )

    @timer
    def run(self):
        """
        Run block-update PMMH.

        Returns same structure as PMMH.run().  accepted[i] is True if at least
        one block proposal was accepted during iteration i.
        """
        rng = self.rng
        d   = len(self.theta0)

        chain        = np.zeros((self.n_iter + 1, d))
        loglik_chain = np.zeros(self.n_iter + 1)
        accepted     = np.zeros(self.n_iter, dtype=bool)

        theta_curr  = self.theta0.copy()
        loglik_curr = self._evaluate_loglik(theta_curr)
        lp_curr     = self._log_prior_and_jac(theta_curr)

        chain[0]        = theta_curr
        loglik_chain[0] = loglik_curr

        for i in range(self.n_iter):
            for block in self.blocks:
                idx        = np.array(block)
                theta_prop = theta_curr.copy()
                theta_prop[idx] += rng.normal(0.0, self.step_sizes[idx])

                try:
                    loglik_prop = self._evaluate_loglik(theta_prop)
                except Exception:
                    continue

                lp_prop   = self._log_prior_and_jac(theta_prop)
                log_alpha = (lp_prop + loglik_prop) - (lp_curr + loglik_curr)

                if np.log(rng.uniform()) < log_alpha:
                    theta_curr  = theta_prop
                    loglik_curr = loglik_prop
                    lp_curr     = lp_prop
                    accepted[i] = True

            chain[i + 1]        = theta_curr
            loglik_chain[i + 1] = loglik_curr

        self.chain        = chain
        self.loglik_chain = loglik_chain
        self.accepted     = accepted
        self.accept_rate  = float(accepted.mean())

        self.model.update_params(self.model.constrain_params(theta_curr))
        return chain, loglik_chain, accepted
