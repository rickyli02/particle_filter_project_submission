# Hamiltonian Monte Carlo for state-space models with a closed-form
# log-likelihood and analytic score.
#
# The model must implement:
#   log_likelihood(data)              — log p(y_{0:T-1} | θ)
#   score(data)                       — ∇_θ log p(y | θ),  shape (d,)
#   constrain_params(unc_vector)      — R^d → constrained θ
#   unconstrain_params(con_params)    — constrained θ → R^d
#   jacobian_constrain_params(u)      — d×d diagonal Jacobian dθ/du
#   update_params(con_params)         — update model in-place
#
# The gradient of the full log-target in unconstrained space is
#
#   ∇_u log π(u) = ∇_u ℓ(θ(u); y)  +  ∇_u [log p(θ(u)) + log|det J(u)|]
#
# The first term is computed exactly via the chain rule
#   ∇_u ℓ = diag(J) ⊙ ∇_θ ℓ          (J is diagonal for our transforms)
# The second term uses central finite differences (prior and Jacobian are cheap).

from __future__ import annotations

import numpy as np

from estimation.mcmc import MCMCBase
from models.base import StateSpaceModel
from utils import timer


class HamiltonianMC(MCMCBase):
    """
    Hamiltonian Monte Carlo for SSMs with a closed-form log-likelihood and score.

    Targets the posterior in unconstrained space using the same
    constrain/unconstrain convention as MetropolisHastings.

    Parameters
    ----------
    model      : StateSpaceModel — must implement log_likelihood and score
    data       : array-like — observations passed to model.log_likelihood / score
    n_iter     : int — number of HMC iterations
    step_size  : float — leapfrog step size ε
    n_leapfrog : int — leapfrog steps L per proposal
    mass_diag  : array of length d, or None
                 Diagonal of the mass matrix M.  Defaults to ones (identity).
                 Tuning tip: set to approximate posterior variances.
    theta0     : array-like of length d — initial *unconstrained* params;
                 use model.unconstrain_params(...) to obtain this
    log_prior  : callable or None — see MCMCBase
    prior_space: {"constrained", "unconstrained"}
    include_jacobian : bool | None
    seed       : int | None
    """

    def __init__(
        self,
        model: StateSpaceModel,
        data,
        n_iter: int = 2000,
        step_size: float = 0.1,
        n_leapfrog: int = 10,
        mass_diag=None,
        theta0=None,
        log_prior=None,
        prior_space: str = "constrained",
        include_jacobian=None,
        seed=None,
    ):
        if not hasattr(model, "log_likelihood"):
            raise ValueError(
                f"{type(model).__name__} does not implement log_likelihood. "
                "HamiltonianMC requires a closed-form log-likelihood."
            )
        if not hasattr(model, "score"):
            raise ValueError(
                f"{type(model).__name__} does not implement score. "
                "HamiltonianMC requires an analytic score."
            )

        d = len(np.asarray(theta0, dtype=float))
        super().__init__(
            model,
            n_iter,
            step_sizes=np.full(d, step_size),
            theta0=theta0,
            log_prior=log_prior,
            prior_space=prior_space,
            include_jacobian=include_jacobian,
            seed=seed,
        )
        self.data       = data
        self.step_size  = float(step_size)
        self.n_leapfrog = int(n_leapfrog)
        self.mass_diag  = (
            np.ones(d) if mass_diag is None
            else np.asarray(mass_diag, dtype=float)
        )

    def __repr__(self) -> str:
        return (
            f"HamiltonianMC(model={self.model!r}, n_iter={self.n_iter}, "
            f"step_size={self.step_size}, n_leapfrog={self.n_leapfrog})"
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _evaluate_loglik(self, theta_unc: np.ndarray) -> float:
        """Constrain θ, update model, and return log p(y | θ)."""
        constrained = self.model.constrain_params(theta_unc)
        self.model.update_params(constrained)
        self.model.clear_state()
        ll = self.model.log_likelihood(self.data)
        return float(ll) if np.isfinite(ll) else -np.inf

    def _grad_log_target(self, theta_unc: np.ndarray) -> np.ndarray:
        """
        Gradient of log π(u) = ℓ(θ(u); y) + log_prior_term(u).

        The log-likelihood part is exact (one Kalman pass via model.score).
        The prior + Jacobian part uses central finite differences (cheap).

        Returns shape-(d,) gradient array.
        """
        # ── exact gradient of log-likelihood ──────────────────────────────────
        constrained = self.model.constrain_params(theta_unc)
        self.model.update_params(constrained)
        score_con = self.model.score(self.data)                     # ∇_θ ℓ
        j_diag    = np.diag(self.model.jacobian_constrain_params(theta_unc))
        grad      = j_diag * score_con                              # ∇_u ℓ

        # ── finite-difference gradient of prior + Jacobian correction ─────────
        h = 1e-5
        d = len(theta_unc)
        for j in range(d):
            e        = np.zeros(d); e[j] = h
            grad[j] += (
                self._log_prior_term(theta_unc + e)
                - self._log_prior_term(theta_unc - e)
            ) / (2.0 * h)

        return grad

    def _leapfrog(
        self, u: np.ndarray, p: np.ndarray, grad_u: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run n_leapfrog leapfrog steps from position u, momentum p.

        The integrator is:
          p ← p + (ε/2) ∇log π(u)          half-kick
          for l = 1 … L:
            u ← u + ε M⁻¹ p                drift
            p ← p + ε ∇log π(u)  (or ε/2 on the last step)

        Returns (u_new, p_new, grad_at_u_new).
        """
        eps      = self.step_size
        mass_inv = 1.0 / self.mass_diag

        u = u.copy()
        p = p.copy()

        # Half-kick with current gradient
        p += 0.5 * eps * grad_u

        for l in range(self.n_leapfrog):
            u      += eps * mass_inv * p
            grad_u  = self._grad_log_target(u)
            if l < self.n_leapfrog - 1:
                p += eps * grad_u          # full kick
            else:
                p += 0.5 * eps * grad_u   # final half-kick

        return u, p, grad_u

    # ── public API ────────────────────────────────────────────────────────────

    @timer
    def run(self):
        """
        Run HMC for n_iter iterations.

        Returns
        -------
        chain        : (n_iter + 1, d) — unconstrained samples; row 0 is theta0
        loglik_chain : (n_iter + 1,)   — log p(y | θ) at each sample
        accepted     : (n_iter,) bool  — True where the proposal was accepted
        """
        rng = self.rng
        d   = len(self.theta0)

        chain        = np.zeros((self.n_iter + 1, d))
        loglik_chain = np.zeros(self.n_iter + 1)
        accepted     = np.zeros(self.n_iter, dtype=bool)

        u_curr      = self.theta0.copy()
        grad_curr   = self._grad_log_target(u_curr)
        loglik_curr = self._evaluate_loglik(u_curr)
        lp_curr     = self._log_prior_term(u_curr)

        chain[0]        = u_curr
        loglik_chain[0] = loglik_curr

        for i in range(self.n_iter):
            # Sample momentum from N(0, M)
            p_curr = rng.normal(0.0, np.sqrt(self.mass_diag), size=d)

            # Propose via leapfrog
            try:
                u_prop, p_prop, grad_prop = self._leapfrog(u_curr, p_curr, grad_curr)
            except Exception:
                chain[i + 1]        = u_curr
                loglik_chain[i + 1] = loglik_curr
                continue

            loglik_prop = self._evaluate_loglik(u_prop)
            lp_prop     = self._log_prior_term(u_prop)

            # Metropolis accept/reject via Hamiltonian difference
            # H(u, p) = -log π(u) + ½ pᵀ M⁻¹ p
            k_curr    = 0.5 * np.sum(p_curr ** 2 / self.mass_diag)
            k_prop    = 0.5 * np.sum(p_prop ** 2 / self.mass_diag)
            log_alpha = (loglik_prop + lp_prop) - (loglik_curr + lp_curr) + k_curr - k_prop

            if np.log(rng.uniform()) < log_alpha:
                u_curr      = u_prop
                grad_curr   = grad_prop
                loglik_curr = loglik_prop
                lp_curr     = lp_prop
                accepted[i] = True

            chain[i + 1]        = u_curr
            loglik_chain[i + 1] = loglik_curr

        self.chain        = chain
        self.loglik_chain = loglik_chain
        self.accepted     = accepted
        self.accept_rate  = float(accepted.mean())

        self.model.update_params(self.model.constrain_params(u_curr))
        return chain, loglik_chain, accepted
