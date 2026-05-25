# Metropolis-Hastings for state-space models with a closed-form log-likelihood.
#
# Structurally identical to PMMH (pmmh.py) but uses model.log_likelihood(data)
# directly instead of a particle filter estimate.  This gives exact posterior
# samples for models where the marginal likelihood is analytically tractable
# (e.g. SimpleLinearGaussianSSM and LinearGaussianSSM via the Kalman filter).
#
# The model must implement:
#     log_likelihood(data)           — log p(y_{0:T-1} | θ)
#     constrain_params(unc_vector)   — R^d → constrained params
#     unconstrain_params(con_params) — constrained params → R^d
#     update_params(con_params)      — update model in-place
#
# BlockMetropolisHastings cycles through parameter subsets independently,
# mirroring BlockPMMH.

from __future__ import annotations

import numpy as np

from models.base import StateSpaceModel


class MetropolisHastings:
    """
    Random-walk Metropolis-Hastings for state-space models with a
    closed-form log-likelihood.

    Targets the posterior p(θ | y) ∝ p(y | θ) · p(θ) exactly.
    Proposals are Gaussian random walks in the *unconstrained* parameter space;
    the model's constrain_params / unconstrain_params transforms map between
    the two spaces.

    Parameters
    ----------
    model      : StateSpaceModel — must implement log_likelihood(data)
    data       : array-like — observations passed to model.log_likelihood
    n_iter     : int — number of MCMC iterations
    step_sizes : array-like of length d, or None (defaults to 0.1 per dim)
    theta0     : array-like of length d — initial *unconstrained* params;
                 use model.unconstrain_params(...) to obtain this
    log_prior  : callable(theta_unc) -> float, or None (flat prior)
    seed       : int | None
    """

    def __init__(
        self,
        model: StateSpaceModel,
        data,
        n_iter: int = 2000,
        step_sizes=None,
        theta0=None,
        log_prior=None,
        seed=None,
    ):
        if not isinstance(model, StateSpaceModel):
            raise ValueError("model must be a StateSpaceModel instance.")
        if not hasattr(model, "log_likelihood"):
            raise ValueError(
                f"{type(model).__name__} does not implement log_likelihood(data). "
                "MetropolisHastings requires a closed-form log-likelihood."
            )
        if theta0 is None:
            raise ValueError(
                "theta0 (initial unconstrained parameter vector) is required. "
                "Use model.unconstrain_params(...) to obtain it."
            )

        self.model     = model
        self.data      = data
        self.n_iter    = n_iter
        self.log_prior = log_prior if log_prior is not None else lambda _: 0.0
        self.rng       = np.random.default_rng(seed)

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

        self.chain        = None
        self.loglik_chain = None
        self.accepted     = None
        self.accept_rate  = None

    def __repr__(self) -> str:
        return (
            f"MetropolisHastings(model={self.model!r}, n_iter={self.n_iter})"
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _evaluate_loglik(self, theta_unc: np.ndarray) -> float:
        """Constrain θ, update model, and return log p(y | θ). Returns -inf on error."""
        constrained = self.model.constrain_params(theta_unc)
        self.model.update_params(constrained)
        self.model.clear_state()
        ll = self.model.log_likelihood(self.data)
        return float(ll) if np.isfinite(ll) else -np.inf

    # ── public API ────────────────────────────────────────────────────────────

    def run(self):
        """
        Run Metropolis-Hastings for n_iter iterations.

        Returns
        -------
        chain        : (n_iter + 1, d)  unconstrained samples; row 0 is theta0
        loglik_chain : (n_iter + 1,)    log p(y | θ) at each sample
        accepted     : (n_iter,) bool   True where the proposal was accepted
        """
        rng = self.rng
        d   = len(self.theta0)

        chain        = np.zeros((self.n_iter + 1, d))
        loglik_chain = np.zeros(self.n_iter + 1)
        accepted     = np.zeros(self.n_iter, dtype=bool)

        theta_curr     = self.theta0.copy()
        loglik_curr    = self._evaluate_loglik(theta_curr)
        log_prior_curr = self.log_prior(theta_curr)

        chain[0]        = theta_curr
        loglik_chain[0] = loglik_curr

        for i in range(self.n_iter):
            theta_prop = theta_curr + rng.normal(0.0, self.step_sizes)

            try:
                loglik_prop = self._evaluate_loglik(theta_prop)
            except Exception:
                chain[i + 1]        = theta_curr
                loglik_chain[i + 1] = loglik_curr
                continue

            log_prior_prop = self.log_prior(theta_prop)
            log_alpha = (log_prior_prop + loglik_prop) - (log_prior_curr + loglik_curr)

            if np.log(rng.uniform()) < log_alpha:
                theta_curr     = theta_prop
                loglik_curr    = loglik_prop
                log_prior_curr = log_prior_prop
                accepted[i]    = True

            chain[i + 1]        = theta_curr
            loglik_chain[i + 1] = loglik_curr

        self.chain        = chain
        self.loglik_chain = loglik_chain
        self.accepted     = accepted
        self.accept_rate  = float(accepted.mean())

        self.model.update_params(self.model.constrain_params(theta_curr))
        return chain, loglik_chain, accepted


class BlockMetropolisHastings(MetropolisHastings):
    """
    Block-update Metropolis-Hastings.

    Each iteration cycles through a set of parameter blocks, proposing and
    accepting / rejecting each block independently while re-evaluating the
    exact log-likelihood for each.  Mirrors BlockPMMH but without a PF.

    Parameters
    ----------
    blocks : list of list of int, or None
             Index groups in the unconstrained vector, e.g. [[0, 2], [1, 3]].
             None defaults to one block per parameter (component-wise MH).
    """

    def __init__(
        self,
        model: StateSpaceModel,
        data,
        n_iter: int = 2000,
        step_sizes=None,
        theta0=None,
        log_prior=None,
        seed=None,
        blocks=None,
    ):
        super().__init__(model, data, n_iter, step_sizes, theta0, log_prior, seed)
        d = len(self.theta0)
        self.blocks = blocks if blocks is not None else [[i] for i in range(d)]

    def __repr__(self) -> str:
        return (
            f"BlockMetropolisHastings(model={self.model!r}, "
            f"n_iter={self.n_iter}, n_blocks={len(self.blocks)})"
        )

    @timer
    def run(self):
        """
        Run block-update MH.

        Returns same structure as MetropolisHastings.run().
        accepted[i] is True if at least one block was accepted during iteration i.
        """
        rng = self.rng
        d   = len(self.theta0)

        chain        = np.zeros((self.n_iter + 1, d))
        loglik_chain = np.zeros(self.n_iter + 1)
        accepted     = np.zeros(self.n_iter, dtype=bool)

        theta_curr     = self.theta0.copy()
        loglik_curr    = self._evaluate_loglik(theta_curr)
        log_prior_curr = self.log_prior(theta_curr)

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

                log_prior_prop = self.log_prior(theta_prop)
                log_alpha = (
                    (log_prior_prop + loglik_prop) - (log_prior_curr + loglik_curr)
                )

                if np.log(rng.uniform()) < log_alpha:
                    theta_curr     = theta_prop
                    loglik_curr    = loglik_prop
                    log_prior_curr = log_prior_prop
                    accepted[i]    = True

            chain[i + 1]        = theta_curr
            loglik_chain[i + 1] = loglik_curr

        self.chain        = chain
        self.loglik_chain = loglik_chain
        self.accepted     = accepted
        self.accept_rate  = float(accepted.mean())

        self.model.update_params(self.model.constrain_params(theta_curr))
        return chain, loglik_chain, accepted
