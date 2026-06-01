# Two-stage Nelder-Mead Particle Marginal Maximum Likelihood Estimator (PMMLE).
#
# Maximises the particle-filter log-likelihood estimate p̂(y | θ) w.r.t. θ.
# The noisy PF objective makes gradient methods unreliable; Nelder-Mead avoids
# gradient computation and is reasonably robust to low-level Monte Carlo noise.
#
# The two-stage design reduces the cost of the expensive high-particle stage:
#
#   Stage 1 (coarse):  n_restarts runs of Nelder-Mead with N_particles_1.
#                      Each restart uses a fixed PF seed (deterministic
#                      objective) but a fresh seed per restart (seed variety
#                      reduces seed-specific bias).  Loose tolerances keep
#                      evaluations fast.
#
#   Stage 2 (fine):    Single Nelder-Mead run from the stage-1 best, using
#                      N_particles_2 and a fresh seed.  Tighter tolerances.
#
# Using a fixed seed per run makes the PF objective deterministic in θ for
# that run via the "common random numbers" trick — Nelder-Mead then converges
# reliably.  The fresh seed in stage 2 ensures the final estimate is not
# overfit to the stage-1 noise realization.

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import LinAlgError

from models.base import StateSpaceModel
from estimation.particle_filter import ParticleFilter
from estimation.resampling_methods import ResamplingMethod, SystematicResampling
from utils import timer


# ── result container ──────────────────────────────────────────────────────────

@dataclass
class PMMResult:
    """
    Returned by NelderMeadPMMLE.fit().

    Attributes
    ----------
    param_names          : list[str]
    constrained_params   : object        — model's native format
    unconstrained_params : np.ndarray    — optimizer's unconstrained space
    loglik               : float         — PF log-lik at the PMLE (fresh seed)
    success_1            : bool          — any stage-1 restart converged
    success_2            : bool          — stage-2 convergence flag
    n_evals_1            : int           — total stage-1 function evaluations
    n_evals_2            : int           — stage-2 function evaluations
    message              : str           — stage-2 optimizer message
    """
    param_names: list
    constrained_params: object
    unconstrained_params: np.ndarray
    loglik: float
    success_1: bool
    success_2: bool
    n_evals_1: int
    n_evals_2: int
    message: str

    def summary(self) -> str:
        lines = [
            f"PMMResult  loglik={self.loglik:.4f}",
            f"  Stage 1: {'converged' if self.success_1 else 'NOT CONVERGED'}  "
            f"n_evals={self.n_evals_1}",
            f"  Stage 2: {'converged' if self.success_2 else 'NOT CONVERGED'}  "
            f"n_evals={self.n_evals_2}",
            f"  {self.message}",
            "",
            f"  {'Parameter':<16}  {'Estimate':>12}",
            "  " + "-" * 32,
        ]
        params = (
            list(self.constrained_params.values())
            if hasattr(self.constrained_params, "items")
            else list(self.constrained_params)
        )
        for name, val in zip(self.param_names, params):
            lines.append(f"  {name:<16}  {val:>12.6f}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PMMResult(loglik={self.loglik:.4f}, "
            f"success=({self.success_1},{self.success_2}), "
            f"params={dict(zip(self.param_names, self._flat_constrained()))})"
        )

    def _flat_constrained(self):
        p = self.constrained_params
        return list(p.values()) if hasattr(p, "items") else list(p)


# ── estimator ─────────────────────────────────────────────────────────────────

class NelderMeadPMMLE:
    """
    Two-stage Particle Marginal Maximum Likelihood Estimator.

    Parameters
    ----------
    model           : StateSpaceModel
        Must implement constrain_params, unconstrain_params, update_params.
    data            : array-like
    N_particles_1   : int   — stage-1 particle count (coarse, smaller)
    N_particles_2   : int   — stage-2 particle count (fine, larger)
    resample_method : ResamplingMethod | None
        Resampling class to use.  Defaults to SystematicResampling.
        A fresh instance with a fixed seed is constructed per PF call.
    n_restarts      : int   — number of independent stage-1 runs
    restart_std     : float — std of Gaussian noise added to theta0 for restarts
    seed            : int | None
    """

    def __init__(
        self,
        model: StateSpaceModel,
        data,
        N_particles_1: int = 200,
        N_particles_2: int = 1000,
        resample_method: ResamplingMethod | None = None,
        filter_cls=None,
        n_restarts: int = 3,
        restart_std: float = 0.5,
        seed=None,
    ):
        if not isinstance(model, StateSpaceModel):
            raise ValueError("model must be a StateSpaceModel instance.")
        self.model = model
        self.data = np.asarray(data)
        self.N_particles_1 = N_particles_1
        self.N_particles_2 = N_particles_2
        self._resample_cls = (
            type(resample_method) if resample_method is not None else SystematicResampling
        )
        # filter_cls: ParticleFilter subclass to use (e.g. RaoBlackwellizedParticleFilter).
        # Defaults to ParticleFilter for standard models.
        self._filter_cls = filter_cls if filter_cls is not None else ParticleFilter
        self.n_restarts = n_restarts
        self.restart_std = restart_std
        self.rng = np.random.default_rng(seed)
        self.result: PMMResult | None = None

    def __repr__(self) -> str:
        return (
            f"NelderMeadPMMLE(model={self.model!r}, "
            f"N1={self.N_particles_1}, N2={self.N_particles_2}, "
            f"n_restarts={self.n_restarts})"
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _pf_loglik(self, theta_unc: np.ndarray, n_particles: int, pf_seed: int) -> float:
        """
        Evaluate the PF log-likelihood at theta_unc.

        The model RNG is reset to pf_seed before each call so that the
        particle draws are identical across calls with the same (theta, pf_seed)
        pair — making the objective deterministic within a stage.
        """
        try:
            constrained = self.model.constrain_params(theta_unc)
            self.model.update_params(constrained)
        except (ValueError, LinAlgError, FloatingPointError):
            return -np.inf

        # Reset model RNG for reproducibility (used by bootstrap PF; harmless for RBPF).
        self.model.rng = np.random.default_rng(pf_seed)

        pf = self._filter_cls(
            model=self.model,
            N_particles=n_particles,
            data=self.data,
            resample_method=self._resample_cls(seed=pf_seed),
            seed=pf_seed,
        )
        try:
            _, _, _, _, loglik = pf.run_filter()
        except Exception:
            return -np.inf
        return float(loglik)

    def _make_objective(self, n_particles: int, pf_seed: int, assemble) -> callable:
        def neg_loglik(u_free: np.ndarray) -> float:
            ll = self._pf_loglik(assemble(u_free), n_particles, pf_seed)
            return -ll if np.isfinite(ll) else np.inf
        return neg_loglik

    @staticmethod
    def _nelder_mead(objective, theta0: np.ndarray, xatol: float, fatol: float):
        return minimize(
            objective,
            theta0,
            method="Nelder-Mead",
            options={
                "maxiter":  20000,
                "maxfev":   20000,
                "xatol":    xatol,
                "fatol":    fatol,
                "adaptive": True,
            },
        )

    def _parse_fixed_params(self, fixed_params: dict | None):
        param_names = list(self.model.params_dict.keys())
        fixed_params = fixed_params or {}
        for name in fixed_params:
            if name not in param_names:
                raise ValueError(
                    f"fixed_params key '{name}' not found in model params: {param_names}"
                )
        free_idx  = [i for i, n in enumerate(param_names) if n not in fixed_params]
        fixed_idx = [i for i, n in enumerate(param_names) if n in fixed_params]
        ref = list(self.model.params)
        for i, name in enumerate(param_names):
            if name in fixed_params:
                ref[i] = fixed_params[name]
        u_ref   = np.asarray(self.model.unconstrain_params(ref), dtype=float)
        u_fixed = u_ref[fixed_idx] if fixed_idx else np.empty(0)
        return param_names, free_idx, fixed_idx, u_fixed

    def _make_assembler(self, n: int, free_idx: list, fixed_idx: list, u_fixed: np.ndarray):
        fi = np.array(free_idx,  dtype=int)
        xi = np.array(fixed_idx, dtype=int)
        def assemble(u_free: np.ndarray) -> np.ndarray:
            full = np.empty(n)
            if fi.size: full[fi] = u_free
            if xi.size: full[xi] = u_fixed
            return full
        return assemble

    # ── public API ────────────────────────────────────────────────────────────

    @timer
    def fit(self, theta0=None, fixed_params: dict | None = None) -> PMMResult:
        """
        Fit by maximising the PF log-likelihood estimate.

        Parameters
        ----------
        theta0 : array-like | None
            Initial unconstrained parameter vector (full dimension).
            Defaults to model.unconstrain_params(model.params).
        fixed_params : dict[str, float] | None
            Parameter names → fixed constrained values.
            E.g. {'alpha': 1.0} keeps alpha fixed.  None frees all params.

        Returns
        -------
        PMMResult
        """
        param_names, free_idx, fixed_idx, u_fixed = self._parse_fixed_params(fixed_params)
        assemble = self._make_assembler(len(param_names), free_idx, fixed_idx, u_fixed)

        if theta0 is None:
            full_theta0 = np.asarray(self.model.unconstrain_params(self.model.params), dtype=float)
        else:
            full_theta0 = np.asarray(theta0, dtype=float)
        theta0_free = full_theta0[free_idx] if free_idx else np.empty(0)

        # ── Stage 1: coarse multi-restart search ─────────────────────────────
        print(f"Stage 1  N_particles={self.N_particles_1}, {self.n_restarts} restart(s)")

        starts = [theta0_free] + [
            theta0_free + self.rng.normal(0.0, self.restart_std, size=theta0_free.shape)
            for _ in range(self.n_restarts - 1)
        ]

        best_1, best_val_1 = None, np.inf
        total_evals_1, any_converged_1 = 0, False

        for i, start in enumerate(starts):
            pf_seed_1 = int(self.rng.integers(0, 2**31))
            obj_1 = self._make_objective(self.N_particles_1, pf_seed_1, assemble)
            opt   = self._nelder_mead(obj_1, start, xatol=1e-3, fatol=5e-2)
            total_evals_1 += opt.nfev
            if opt.success:
                any_converged_1 = True
            if opt.fun < best_val_1:
                best_val_1, best_1 = opt.fun, opt
            print(f"  restart {i+1}/{self.n_restarts}: "
                  f"loglik≈{-opt.fun:.2f}  nfev={opt.nfev}  "
                  f"{'converged' if opt.success else 'not converged'}")

        # ── Stage 2: fine refinement from stage-1 best ───────────────────────
        print(f"\nStage 2  N_particles={self.N_particles_2}, starting from stage-1 best")

        pf_seed_2 = int(self.rng.integers(0, 2**31))
        obj_2 = self._make_objective(self.N_particles_2, pf_seed_2, assemble)
        opt_2 = self._nelder_mead(obj_2, best_1.x, xatol=1e-4, fatol=1e-3)

        print(f"  loglik≈{-opt_2.fun:.2f}  nfev={opt_2.nfev}  "
              f"{'converged' if opt_2.success else 'not converged'}")

        # ── Final log-lik estimate with a fresh seed ──────────────────────────
        # Avoids reporting a value overfit to the stage-2 noise realization.
        theta_mle_free = opt_2.x
        full_theta_mle = assemble(theta_mle_free)
        constrained    = self.model.constrain_params(full_theta_mle)
        self.model.update_params(constrained)

        final_seed = int(self.rng.integers(0, 2**31))
        final_ll   = self._pf_loglik(full_theta_mle, self.N_particles_2, final_seed)

        self.result = PMMResult(
            param_names=param_names,
            constrained_params=constrained,
            unconstrained_params=full_theta_mle,
            loglik=final_ll,
            success_1=any_converged_1,
            success_2=opt_2.success,
            n_evals_1=total_evals_1,
            n_evals_2=opt_2.nfev,
            message=opt_2.message,
        )
        return self.result