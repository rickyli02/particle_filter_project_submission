"""
Abstract base class for MCMC samplers over state-space model parameters.

All concrete samplers target a posterior in unconstrained space z. When the
user supplies a prior on constrained parameters θ = θ(z), the samplers include
the usual change-of-variables correction:

    π(z) ∝ p(y | θ(z)) · p_prior(θ(z)) · |det J(z)|

where z is the unconstrained parameter vector, θ(z) = constrain_params(z), and
J(z) = dθ/dz is the Jacobian of the constrain_params transform.

If instead the user supplies a prior directly on z, the target becomes

    π(z) ∝ p(y | θ(z)) · p_prior(z)

Subclasses must implement:
    _evaluate_loglik(theta_unc) → float   log p(y|θ) or an unbiased estimate
    run()                                 main sampling loop
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from models.base import StateSpaceModel


class MCMCBase(ABC):
    """
    Abstract base for random-walk MCMC samplers over SSM parameters.

    Parameters
    ----------
    model      : StateSpaceModel
    n_iter     : int — number of MCMC iterations
    step_sizes : array-like of length d, or None (defaults to 0.1 per dim)
    theta0     : array-like of length d — initial *unconstrained* parameter vector;
                 obtain via model.unconstrain_params(constrained_params)
    log_prior  : callable, or None
                 Log prior. Interpreted according to prior_space.
    prior_space: {"constrained", "unconstrained"}
                 Whether log_prior is defined on constrained θ or unconstrained z.
    include_jacobian : bool | None
                 Whether to add log|det dθ/dz| to the target.
                 Default: True for constrained priors, False for unconstrained priors.
    seed       : int | None
    """

    def __init__(
        self,
        model: StateSpaceModel,
        n_iter: int,
        step_sizes,
        theta0,
        log_prior,
        prior_space,
        include_jacobian,
        seed,
    ):
        if not isinstance(model, StateSpaceModel):
            raise ValueError("model must be a StateSpaceModel instance.")
        if theta0 is None:
            raise ValueError(
                "theta0 (initial unconstrained parameter vector) is required. "
                "Use model.unconstrain_params(...) to obtain it."
            )
        if prior_space not in {"constrained", "unconstrained"}:
            raise ValueError(
                "prior_space must be either 'constrained' or 'unconstrained'."
            )

        self.model         = model
        self.n_iter        = n_iter
        self.log_prior     = log_prior if log_prior is not None else lambda _: 0.0
        self.prior_space   = prior_space
        self.include_jacobian = (
            prior_space == "constrained"
            if include_jacobian is None
            else bool(include_jacobian)
        )
        self.seed          = seed
        self.rng           = np.random.default_rng(seed)

        self.theta0        = np.asarray(theta0, dtype=float)
        d                  = len(self.theta0)
        self.step_sizes    = (
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

    # ── abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def _evaluate_loglik(self, theta_unc: np.ndarray) -> float:
        """
        Evaluate log p(y | θ(z)), or an unbiased estimate thereof.

        Must constrain theta_unc, update the model, and return a scalar.
        Must return -inf for any proposal that produces an invalid model.
        """
        ...

    @abstractmethod
    def run(self):
        """
        Run the sampler for n_iter iterations.

        Must populate self.chain, self.loglik_chain, self.accepted,
        self.accept_rate and return (chain, loglik_chain, accepted).

        chain        : (n_iter + 1, d)  unconstrained samples; row 0 is theta0
        loglik_chain : (n_iter + 1,)    log p(y | θ) at each sample
        accepted     : (n_iter,) bool   True where the proposal was accepted
        """
        ...

    # ── shared internals ──────────────────────────────────────────────────────

    def _log_abs_det_jacobian(self, theta_unc: np.ndarray) -> float:
        """
        log|det dθ/dz| for the model's constrain_params transform.

        Supports both diagonal Jacobians and general square Jacobians. Raises
        NotImplementedError if the model does not provide jacobian_constrain_params.
        """
        J = np.asarray(self.model.jacobian_constrain_params(theta_unc), dtype=float)
        if J.ndim != 2 or J.shape[0] != J.shape[1]:
            raise ValueError(
                "jacobian_constrain_params must return a square 2-D array."
            )
        sign, logabsdet = np.linalg.slogdet(J)
        if sign == 0:
            return -np.inf
        return float(logabsdet)

    def _log_prior_term(self, theta_unc: np.ndarray) -> float:
        """
        Prior contribution to the target density in unconstrained space z.

        - For constrained priors: log p(θ(z)) [+ log|det dθ/dz| if requested]
        - For unconstrained priors: log p(z)
        """
        if self.prior_space == "constrained":
            lp = self.log_prior(self.model.constrain_params(theta_unc))
        else:
            lp = self.log_prior(theta_unc)

        if self.include_jacobian:
            lp += self._log_abs_det_jacobian(theta_unc)

        return float(lp)

    def _log_prior_and_jac(self, theta_unc: np.ndarray) -> float:
        """
        Backward-compatible alias for the target prior contribution.
        """
        return self._log_prior_term(theta_unc)

    # ── shared post-run API ───────────────────────────────────────────────────

    @property
    def constrained_chain(self) -> np.ndarray:
        """Constrained-space version of the post-run chain. Requires run() first."""
        if self.chain is None:
            raise RuntimeError(
                f"Call {type(self).__name__}.run() before accessing constrained_chain."
            )
        return self.model.constrain_chain(self.chain)

    def summary(self, burn: int = 0) -> None:
        """Print posterior mean, std, and acceptance rate for each constrained parameter."""
        if self.chain is None:
            raise RuntimeError(
                f"Call {type(self).__name__}.run() before calling summary()."
            )
        con   = self.model.constrain_chain(self.chain[burn:])
        names = list(self.model.params_dict.keys())
        print(f"{'param':<12} {'mean':>10} {'std':>10}")
        print("-" * 34)
        for j, name in enumerate(names):
            print(f"{name:<12} {con[:, j].mean():>10.4f} {con[:, j].std():>10.4f}")
        print(f"\nAcceptance rate: {self.accept_rate:.3f}  (burn={burn})")
