# Maximum likelihood estimation for state-space models.
#
# Uses the model's closed-form log_likelihood(data) when available (e.g.
# SimpleLinearGaussianSSM and LinearGaussianSSM via the Kalman filter).
#
# Can also be applied to approximate log-likelihoods (e.g. Kim filter,
# Rao-Blackwellized PF), but the resulting estimates are only as good as the
# approximation — verify in the notebook before trusting SEs or inference.

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import LinAlgError

from models.base import StateSpaceModel

from utils import timer


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class MLEResult:
    """
    Returned by MLEEstimator.fit().

    Attributes
    ----------
    param_names        : list[str]        — ordered names matching params_dict
    constrained_params : object           — in the model's native format
    unconstrained_params : np.ndarray     — in the optimizer's unconstrained space
    loglik             : float            — log p(y | θ_MLE)
    success            : bool             — optimizer convergence flag
    n_evals            : int              — number of objective evaluations
    message            : str             — optimizer status message
    std_errors         : np.ndarray|None  — per-parameter SE in *constrained* space
                                           (set by MLEEstimator.compute_std_errors())
    """
    param_names: list
    constrained_params: object
    unconstrained_params: np.ndarray
    loglik: float
    success: bool
    n_evals: int
    message: str
    std_errors: np.ndarray = field(default=None)

    def summary(self) -> str:
        lines = [
            f"MLEResult  loglik={self.loglik:.4f}  "
            f"{'converged' if self.success else 'NOT CONVERGED'}  "
            f"n_evals={self.n_evals}",
            f"  {self.message}",
            "",
            f"  {'Parameter':<16}  {'Estimate':>12}  {'Std error':>12}",
            "  " + "-" * 44,
        ]
        params = (
            list(self.constrained_params)
            if not hasattr(self.constrained_params, 'items')
            else list(self.constrained_params.values())
        )
        for i, (name, val) in enumerate(zip(self.param_names, params)):
            se = (
                f"{self.std_errors[i]:>12.6f}"
                if self.std_errors is not None and not np.isnan(self.std_errors[i])
                else f"{'—':>12}"
            )
            lines.append(f"  {name:<16}  {val:>12.6f}  {se}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MLEResult(loglik={self.loglik:.4f}, "
            f"success={self.success}, params={dict(zip(self.param_names, self._flat_constrained()))})"
        )

    def _flat_constrained(self):
        params = self.constrained_params
        if hasattr(params, 'items'):
            return list(params.values())
        return list(params)


# ── estimator ─────────────────────────────────────────────────────────────────

class MLEEstimator:
    """
    Maximum-likelihood estimator for state-space models.

    The model must implement:
        log_likelihood(data)          — returns log p(y_{0:T-1} | θ)
        constrain_params(unc_vector)  — maps R^d → constrained params
        unconstrain_params(con_params)— maps constrained params → R^d
        update_params(con_params)     — updates model in-place

    Optimization is carried out in the *unconstrained* space (transforms
    such as tanh for bounded scalars, log for positive scalars). The result
    is reported in the original constrained space.

    Parameters
    ----------
    model       : StateSpaceModel
    data        : array-like — observations passed directly to log_likelihood
    method      : str — scipy optimizer name (default 'L-BFGS-B')
    n_restarts  : int — number of random restarts in addition to the initial
                  start; the best optimum across all runs is returned
    restart_std : float — std of Gaussian perturbations for random restarts
    seed        : int | None
    """

    def __init__(
        self,
        model: StateSpaceModel,
        data,
        method: str = "L-BFGS-B",
        n_restarts: int = 1,
        restart_std: float = 0.5,
        seed=None,
    ):
        if not isinstance(model, StateSpaceModel):
            raise ValueError("model must be a StateSpaceModel instance.")
        if not hasattr(model, "log_likelihood"):
            raise ValueError(
                f"{type(model).__name__} does not implement log_likelihood(data). "
                "MLEEstimator requires a closed-form or approximate log-likelihood."
            )
        self.model = model
        self.data = data
        self.method = method
        self.n_restarts = n_restarts
        self.restart_std = restart_std
        self.rng = np.random.default_rng(seed)
        self.result: MLEResult | None = None

    def __repr__(self) -> str:
        return (
            f"MLEEstimator(model={self.model!r}, method={self.method!r}, "
            f"n_restarts={self.n_restarts})"
        )

    # ── internal ──────────────────────────────────────────────────────────────

    def _neg_loglik(self, theta_unc: np.ndarray) -> float:
        try:
            constrained = self.model.constrain_params(theta_unc)
            self.model.update_params(constrained)
            return -float(self.model.log_likelihood(self.data))
        except (ValueError, LinAlgError, FloatingPointError):
            return np.inf

    def _run_once(self, theta0: np.ndarray):
        return minimize(
            self._neg_loglik,
            theta0,
            method=self.method,
            options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
        )

    # ── public API ────────────────────────────────────────────────────────────
    @timer
    def fit(self, theta0=None) -> MLEResult:
        """
        Fit the model by maximizing log_likelihood.

        Parameters
        ----------
        theta0 : array-like | None
            Initial unconstrained parameter vector.  If None, uses
            model.unconstrain_params(model.params) (current model state).

        Returns
        -------
        MLEResult
        """
        if theta0 is None:
            theta0 = self.model.unconstrain_params(self.model.params)
        theta0 = np.asarray(theta0, dtype=float)

        best_opt = None
        best_val = np.inf

        starts = [theta0] + [
            theta0 + self.rng.normal(0.0, self.restart_std, size=theta0.shape)
            for _ in range(self.n_restarts - 1)
        ]

        for start in starts:
            opt = self._run_once(start)
            if opt.fun < best_val:
                best_val = opt.fun
                best_opt = opt

        theta_mle = best_opt.x
        constrained = self.model.constrain_params(theta_mle)
        self.model.update_params(constrained)

        self.result = MLEResult(
            param_names=list(self.model.params_dict.keys()),
            constrained_params=constrained,
            unconstrained_params=theta_mle,
            loglik=-best_val,
            success=best_opt.success,
            n_evals=best_opt.nfev,
            message=best_opt.message,
        )
        return self.result

    def compute_std_errors(self, eps: float = 1e-4) -> np.ndarray:
        """
        Estimate parameter standard errors from the numerical Hessian at the MLE.

        Standard errors are returned in the *constrained* parameter space via
        the delta method:  SE_constrained ≈ |J| * SE_unconstrained,
        where J is the numerical Jacobian of constrain_params at theta_mle.

        Parameters
        ----------
        eps : float — finite-difference step size

        Returns
        -------
        std_errors : (d,) array in constrained space, NaN where Hessian is
                     singular or the transform is degenerate
        """
        if self.result is None:
            raise RuntimeError("Call fit() before compute_std_errors().")

        theta_mle = self.result.unconstrained_params
        d = len(theta_mle)

        # ── numerical Hessian of the negative log-likelihood ──────────────────
        hess = np.zeros((d, d))
        for i in range(d):
            for j in range(i, d):
                ei = np.zeros(d); ei[i] = eps
                ej = np.zeros(d); ej[j] = eps
                h = (
                    self._neg_loglik(theta_mle + ei + ej)
                    - self._neg_loglik(theta_mle + ei - ej)
                    - self._neg_loglik(theta_mle - ei + ej)
                    + self._neg_loglik(theta_mle - ei - ej)
                ) / (4.0 * eps ** 2)
                hess[i, j] = hess[j, i] = h

        try:
            cov_unc = np.linalg.inv(hess)
        except LinAlgError:
            cov_unc = np.full((d, d), np.nan)

        # ── delta method: transform SE to constrained space ───────────────────
        jac = np.zeros((d, d))
        c0 = np.asarray(
            list(self.model.constrain_params(theta_mle)),
            dtype=float,
        )
        for i in range(d):
            ei = np.zeros(d); ei[i] = eps
            ci = np.asarray(
                list(self.model.constrain_params(theta_mle + ei)),
                dtype=float,
            )
            jac[:, i] = (ci - c0) / eps

        # SE_constrained[k] ≈ sqrt( Jac[k,:] @ cov_unc @ Jac[k,:].T )
        # Non-positive variance means the Hessian is (near-)singular in that direction
        # (e.g. identification ridge) — return nan rather than 0 to make it visible.
        se_con = np.array([
            np.sqrt(v) if (v := jac[k] @ cov_unc @ jac[k]) > 0 else np.nan
            for k in range(d)
        ])

        self.result.std_errors = se_con
        return se_con