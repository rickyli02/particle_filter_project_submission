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

    def _parse_fixed_params(self, fixed_params):
        """Validate fixed_params and return (param_names, free_idx, fixed_idx, u_fixed)."""
        param_names = list(self.model.params_dict.keys())
        fixed_params = fixed_params or {}
        for name in fixed_params:
            if name not in param_names:
                raise ValueError(
                    f"fixed_params key '{name}' not in model params: {param_names}"
                )
        free_idx  = [i for i, n in enumerate(param_names) if n not in fixed_params]
        fixed_idx = [i for i, n in enumerate(param_names) if n in fixed_params]
        # Unconstrain the fixed values by substituting them into a reference vector.
        ref = list(self.model.params)
        for i, name in enumerate(param_names):
            if name in fixed_params:
                ref[i] = fixed_params[name]
        u_ref   = np.asarray(self.model.unconstrain_params(ref), dtype=float)
        u_fixed = u_ref[fixed_idx] if fixed_idx else np.empty(0)
        return param_names, free_idx, fixed_idx, u_fixed

    def _make_assembler(self, n, free_idx, fixed_idx, u_fixed):
        """Return a closure that reconstructs a full unconstrained vector from free params."""
        fi = np.array(free_idx,  dtype=int)
        xi = np.array(fixed_idx, dtype=int)
        def assemble(u_free: np.ndarray) -> np.ndarray:
            full = np.empty(n)
            if fi.size: full[fi] = u_free
            if xi.size: full[xi] = u_fixed
            return full
        return assemble

    def _run_restarts(self, objective, theta0_free: np.ndarray):
        """Run the optimizer with random restarts; return the best scipy result."""
        starts = [theta0_free] + [
            theta0_free + self.rng.normal(0.0, self.restart_std, size=theta0_free.shape)
            for _ in range(self.n_restarts - 1)
        ]
        best_opt, best_val = None, np.inf
        for start in starts:
            opt = minimize(objective, start, method=self.method,
                           options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8})
            if opt.fun < best_val:
                best_val, best_opt = opt.fun, opt
            if start % 10 == 0 and start > 0:
                print(f"{start+1} out of {len(starts)} done...")
        return best_opt

    # ── public API ────────────────────────────────────────────────────────────
    @timer
    def fit(self, theta0=None, fixed_params=None) -> MLEResult:
        """
        Fit the model by maximizing log_likelihood.

        Parameters
        ----------
        theta0 : array-like | None
            Initial unconstrained parameter vector (full dimension).
            If None, uses model.unconstrain_params(model.params).
        fixed_params : dict[str, float] | None
            Map from parameter name to its fixed *constrained* value.
            Names must be keys of model.params_dict.
            E.g. {'alpha': 1.0} holds alpha fixed throughout optimization.
            Parameters not listed here are optimized freely.
            When None (default), all parameters are free.

        Returns
        -------
        MLEResult
        """
        param_names, free_idx, fixed_idx, u_fixed = self._parse_fixed_params(fixed_params)
        assemble = self._make_assembler(len(param_names), free_idx, fixed_idx, u_fixed)

        def neg_loglik_free(u_free: np.ndarray) -> float:
            try:
                self.model.update_params(self.model.constrain_params(assemble(u_free)))
                return -float(self.model.log_likelihood(self.data))
            except (ValueError, LinAlgError, FloatingPointError):
                return np.inf

        self._free_indices       = free_idx
        self._fixed_indices      = fixed_idx
        self._assemble_full      = assemble
        self._neg_loglik_free_fn = neg_loglik_free

        if theta0 is None:
            full_theta0 = np.asarray(self.model.unconstrain_params(self.model.params), dtype=float)
        else:
            full_theta0 = np.asarray(theta0, dtype=float)
        theta0_free = full_theta0[free_idx] if free_idx else np.empty(0)

        best_opt       = self._run_restarts(neg_loglik_free, theta0_free)
        full_theta_mle = assemble(best_opt.x)
        constrained    = self.model.constrain_params(full_theta_mle)
        self.model.update_params(constrained)

        self.result = MLEResult(
            param_names=param_names,
            constrained_params=constrained,
            unconstrained_params=full_theta_mle,
            loglik=-best_opt.fun,
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

        When fit() was called with fixed_params, the Hessian is computed only
        over the free parameters; fixed parameters receive NaN standard errors.

        Parameters
        ----------
        eps : float — finite-difference step size

        Returns
        -------
        std_errors : (n_params,) array in constrained space.
                     NaN for fixed parameters and where the Hessian is singular.
        """
        if self.result is None:
            raise RuntimeError("Call fit() before compute_std_errors().")

        n_params      = len(self.result.unconstrained_params)
        free_indices  = self._free_indices
        fixed_indices = self._fixed_indices
        assemble_fn   = self._assemble_full
        neg_loglik_fn = self._neg_loglik_free_fn

        theta_mle_free = self.result.unconstrained_params[free_indices]
        d = len(theta_mle_free)

        # ── numerical Hessian of the free-param negative log-likelihood ───────
        hess = np.zeros((d, d))
        for i in range(d):
            for j in range(i, d):
                ei = np.zeros(d); ei[i] = eps
                ej = np.zeros(d); ej[j] = eps
                h = (
                    neg_loglik_fn(theta_mle_free + ei + ej)
                    - neg_loglik_fn(theta_mle_free + ei - ej)
                    - neg_loglik_fn(theta_mle_free - ei + ej)
                    + neg_loglik_fn(theta_mle_free - ei - ej)
                ) / (4.0 * eps ** 2)
                hess[i, j] = hess[j, i] = h

        try:
            cov_unc = np.linalg.inv(hess)
        except LinAlgError:
            cov_unc = np.full((d, d), np.nan)

        # ── delta method: Jacobian of constrain_params w.r.t. free unconstrained params
        # Rows: all n_params constrained outputs; columns: d free unconstrained inputs.
        full_mle = assemble_fn(theta_mle_free)
        c0 = np.asarray(list(self.model.constrain_params(full_mle)), dtype=float)
        jac = np.zeros((n_params, d))
        for i in range(d):
            ei = np.zeros(d); ei[i] = eps
            ci = np.asarray(
                list(self.model.constrain_params(assemble_fn(theta_mle_free + ei))),
                dtype=float,
            )
            jac[:, i] = (ci - c0) / eps

        # SE_constrained[k] ≈ sqrt( Jac[k,:] @ cov_unc @ Jac[k,:].T )
        # Fixed parameters and singular Hessian directions return NaN.
        se_con = np.full(n_params, np.nan)
        for k in range(n_params):
            if k not in fixed_indices:
                v = jac[k] @ cov_unc @ jac[k]
                se_con[k] = np.sqrt(v) if v > 0 else np.nan

        self.result.std_errors = se_con
        return se_con