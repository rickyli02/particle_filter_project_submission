# Simple 1-D regime-switching linear Gaussian SSM.
#
# The regime s_t ∈ {0,...,K-1} follows a K-state Markov chain.
# Only process noise is regime-specific; phi, alpha, and tau2 are shared.
#
#   Transition:  x_t = phi * x_{t-1} + eps_t,   eps_t ~ N(0, sigma2_{s_t})
#   Observation: y_t = alpha * x_t   + nu_t,    nu_t  ~ N(0, tau2)
#   Regime:      s_t | s_{t-1} ~ Categorical(P[s_{t-1}, :])
#
# Inherits from RegimeSwitchingBase so the RBPF and KimFilter accept it directly.
#
# Flat parameter layout (length = 3 + K + K^2):
#   constrained  : [phi, alpha, sigma2_0, ..., sigma2_{K-1}, tau2,
#                   P_00, ..., P_{K-1,K-1}]          (P row-major)
#   unconstrained: [arctanh(phi), alpha, log(sigma2_0), ..., log(sigma2_{K-1}),
#                   log(tau2), a_00, ..., a_{K-1,K-1}]
#                  where P[i,j] = softmax(a_i)[j]

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from models.regime_switching_base import (
    RegimeSwitchingBase,
    RegimeSwitchingDims,
    RegimeSwitchingStructure,
)


class SimpleRegimeSwitchingSSM(RegimeSwitchingBase):
    """
    Simple 1-D regime-switching linear Gaussian SSM.

    Parameters
    ----------
    phi        : float      — AR(1) coefficient (require |phi| < 1)
    alpha      : float      — observation loading
    sigma2     : (K,) array — per-regime process noise variances (all > 0)
    tau2       : float      — observation noise variance (shared, > 0)
    trans_matrix : (K, K)  — row-stochastic Markov transition matrix
    seed       : int | None
    """

    def __init__(
        self,
        phi: float,
        alpha: float,
        sigma2,
        tau2: float,
        trans_matrix,
        seed=None,
    ):
        sigma2       = np.asarray(sigma2, dtype=float).ravel()
        trans_matrix = np.asarray(trans_matrix, dtype=float)
        K            = len(sigma2)

        if abs(phi) >= 1:
            raise ValueError(f"phi={phi}: require |phi| < 1 for stationarity.")
        if np.any(sigma2 <= 0):
            raise ValueError("All sigma2 values must be positive.")
        if tau2 <= 0:
            raise ValueError(f"tau2={tau2}: must be positive.")

        dims = RegimeSwitchingDims(n_regimes=K, state_dim=1, obs_dim=1)
        structure = RegimeSwitchingStructure(
            regime_specific_A=False,
            regime_specific_C=False,
            regime_specific_Q=True,
            regime_specific_R=False,
            has_state_intercept=False,
            has_obs_intercept=False,
        )

        super().__init__(
            dims      = dims,
            structure = structure,
            A         = np.array([[phi]]),
            C         = np.array([[alpha]]),
            Q         = [np.array([[s2]]) for s2 in sigma2],
            R         = np.array([[tau2]]),
            P_trans   = trans_matrix,
            seed      = seed,
        )

        # Replace the base class matrix-centric params_dict with scalar names.
        self.params_dict = self._scalar_params_dict()

    # ── scalar properties ─────────────────────────────────────────────────────

    @property
    def phi(self) -> float:
        return float(self.A_list[0][0, 0])

    @property
    def alpha(self) -> float:
        return float(self.C_list[0][0, 0])

    @property
    def sigma2(self) -> np.ndarray:
        return np.array([float(self.Q_list[k][0, 0]) for k in range(self.n_regimes)])

    @property
    def tau2(self) -> float:
        return float(self.R_list[0][0, 0])

    @property
    def stationary_var(self) -> np.ndarray:
        """Per-regime stationary variance sigma2_k / (1 - phi^2)."""
        return self.sigma2 / (1.0 - self.phi ** 2)

    def _scalar_params_dict(self) -> dict:
        d = {"phi": self.phi, "alpha": self.alpha}
        for k, s2 in enumerate(self.sigma2):
            d[f"sigma2_{k}"] = float(s2)
        d["tau2"] = self.tau2
        return d

    # ── repr / describe ───────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"SimpleRegimeSwitchingSSM("
            f"n_regimes={self.n_regimes}, phi={self.phi!r}, "
            f"alpha={self.alpha!r}, sigma2={self.sigma2.tolist()}, "
            f"tau2={self.tau2!r})"
        )

    def describe(self) -> str:
        lines = [
            f"{self.__class__.__name__}",
            f"  1-D regime-switching linear Gaussian SSM",
            f"  Regimes: {self.n_regimes}  "
            f"(sigma2 is regime-specific; phi, alpha, tau2 shared)",
            f"  Transition:  x_t = {self.phi} * x_{{t-1}} + eps_t,  "
            f"eps_t ~ N(0, sigma2_{{s_t}})",
            f"  Observation: y_t = {self.alpha} * x_t + nu_t,  "
            f"nu_t ~ N(0, {self.tau2})",
            f"  sigma2 per regime: {self.sigma2.tolist()}",
            f"  Stationary var per regime: "
            f"{np.round(self.stationary_var, 4).tolist()}",
            f"  Stationary regime probs: "
            f"{np.round(self.regime_probabilities_stationary, 4).tolist()}",
            f"  Regime transition matrix:\n{self.regime_transition_matrix}",
        ]
        return "\n".join(lines)

    # ── parameter interface ───────────────────────────────────────────────────
    # Flat layout: [phi, alpha, sigma2_0,...,sigma2_{K-1}, tau2, P row-major]

    def _n_params(self) -> int:
        K = self.n_regimes
        return 3 + K + K * K   # phi, alpha, K sigma2s, tau2, K×K trans

    def _unpack(self, v) -> tuple:
        K = self.n_regimes
        v = list(v)
        phi    = v[0]
        alpha  = v[1]
        sigma2 = np.array(v[2:2 + K])
        tau2   = v[2 + K]
        P      = np.array(v[3 + K:3 + K + K * K]).reshape(K, K)
        return phi, alpha, sigma2, tau2, P

    def update_params(self, constrained_params):
        phi, alpha, sigma2, tau2, P = self._unpack(constrained_params)
        super().update_params({
            "A": np.array([[phi]]),
            "C": np.array([[alpha]]),
            "Q": [np.array([[s2]]) for s2 in sigma2],
            "R": np.array([[tau2]]),
            "regime_transition_matrix": P,
        })
        self.params_dict = self._scalar_params_dict()

    def unconstrain_params(self, constrained_params) -> np.ndarray:
        """
        Constrained → unconstrained.

        phi      → arctanh(phi)
        sigma2_k → log(sigma2_k)
        tau2     → log(tau2)
        P[i,:]   → log(P[i,:])   (softmax intercepts; P = softmax(log P) since rows sum to 1)
        """
        phi, alpha, sigma2, tau2, P = self._unpack(constrained_params)
        return np.concatenate([
            [np.arctanh(phi)],
            [alpha],
            np.log(sigma2),
            [np.log(tau2)],
            np.log(np.clip(P, 1e-300, None)).ravel(),
        ])

    def constrain_params(self, unconstrained_params) -> list:
        """Unconstrained → constrained (flat list matching update_params)."""
        K = self.n_regimes
        u = np.asarray(unconstrained_params, dtype=float)

        phi    = float(np.tanh(u[0]))
        alpha  = float(u[1])
        sigma2 = np.exp(u[2:2 + K])
        tau2   = float(np.exp(u[2 + K]))

        log_intercepts = u[3 + K:3 + K + K * K].reshape(K, K)
        # softmax row-wise to recover a row-stochastic matrix
        log_intercepts -= log_intercepts.max(axis=1, keepdims=True)   # numerical stability
        P = np.exp(log_intercepts)
        P /= P.sum(axis=1, keepdims=True)

        return ([phi, alpha]
                + sigma2.tolist()
                + [tau2]
                + P.ravel().tolist())

    def jacobian_constrain_params(self, unconstrained_params) -> np.ndarray:
        """
        Diagonal Jacobian of constrain_params (approximate for the transition matrix).

        Exact for phi, sigma2, tau2.  For the transition matrix, uses the diagonal
        of the softmax Jacobian: d(p_ij)/d(a_ij) ≈ p_ij * (1 - p_ij).
        """
        K = self.n_regimes
        u = np.asarray(unconstrained_params, dtype=float)

        diag = np.empty(len(u))
        diag[0] = 1.0 - np.tanh(u[0]) ** 2          # phi
        diag[1] = 1.0                                  # alpha
        diag[2:2 + K] = np.exp(u[2:2 + K])            # sigma2_k
        diag[2 + K] = np.exp(u[2 + K])                # tau2

        log_intercepts = u[3 + K:3 + K + K * K].reshape(K, K)
        log_intercepts -= log_intercepts.max(axis=1, keepdims=True)
        P_rows = np.exp(log_intercepts)
        P_rows /= P_rows.sum(axis=1, keepdims=True)
        diag[3 + K:] = (P_rows * (1.0 - P_rows)).ravel()

        return np.diag(diag)


# ── Fixed-alpha variant ───────────────────────────────────────────────────────

class FixedAlphaRS(SimpleRegimeSwitchingSSM):
    """
    SimpleRegimeSwitchingSSM with alpha fixed at a user-supplied value.

    Removes alpha from the parameter interface so MCMC / MLE only estimates
    (phi, sigma2_0, ..., sigma2_{K-1}, tau2, P).

    Flat layout (length = 2 + K + K^2):
        constrained  : [phi, sigma2_0, ..., sigma2_{K-1}, tau2,
                        P_00, ..., P_{K-1,K-1}]
        unconstrained: [arctanh(phi), log(sigma2_0), ..., log(sigma2_{K-1}),
                        log(tau2), log(P_00), ..., log(P_{K-1,K-1})]
    """

    def __init__(self, alpha_fixed: float, **kwargs):
        self._alpha_fixed = float(alpha_fixed)
        super().__init__(alpha=alpha_fixed, **kwargs)
        self.params_dict = self._fixed_params_dict()

    def __repr__(self) -> str:
        return (
            f"FixedAlphaRS(n_regimes={self.n_regimes}, alpha_fixed={self._alpha_fixed}, "
            f"phi={self.phi!r}, sigma2={self.sigma2.tolist()}, tau2={self.tau2!r})"
        )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _fixed_params_dict(self) -> dict:
        """
        Scalar params exposed for MCMC/MLE.

        Layout matches constrain_params output:
            phi, sigma2_0,...,sigma2_{K-1}, tau2, P_00,...,P_{K-1,K-1}
        """
        d = {'phi': self.phi}
        for k, s2 in enumerate(self.sigma2):
            d[f'sigma2_{k}'] = float(s2)
        d['tau2'] = self.tau2
        P = self.regime_transition_matrix
        for i in range(self.n_regimes):
            for j in range(self.n_regimes):
                d[f'P_{i}{j}'] = float(P[i, j])
        return d

    def _to_full(self, flat) -> list:
        """Insert the fixed alpha to reconstruct the parent's flat layout."""
        flat = list(flat)
        return [flat[0], self._alpha_fixed] + flat[1:]

    def _unpack_fixed(self, flat) -> tuple:
        """Unpack without alpha: (phi, sigma2, tau2, P)."""
        K    = self.n_regimes
        flat = list(flat)
        phi    = flat[0]
        sigma2 = np.array(flat[1:1 + K])
        tau2   = flat[1 + K]
        P      = np.array(flat[2 + K:2 + K + K * K]).reshape(K, K)
        return phi, sigma2, tau2, P

    # ── parameter interface ───────────────────────────────────────────────────

    def update_params(self, flat):
        super().update_params(self._to_full(flat))
        self.params_dict = self._fixed_params_dict()

    def unconstrain_params(self, flat) -> np.ndarray:
        phi, sigma2, tau2, P = self._unpack_fixed(flat)
        return np.concatenate([
            [np.arctanh(phi)],
            np.log(sigma2),
            [np.log(tau2)],
            np.log(np.clip(P, 1e-300, None)).ravel(),
        ])

    def constrain_params(self, unconstrained_params) -> list:
        K = self.n_regimes
        u = np.asarray(unconstrained_params, dtype=float)

        phi    = float(np.tanh(u[0]))
        sigma2 = np.exp(u[1:1 + K])
        tau2   = float(np.exp(u[1 + K]))

        log_p  = u[2 + K:2 + K + K * K].reshape(K, K)
        log_p -= log_p.max(axis=1, keepdims=True)
        P      = np.exp(log_p) / np.exp(log_p).sum(axis=1, keepdims=True)

        return [phi] + sigma2.tolist() + [tau2] + P.ravel().tolist()

    def jacobian_constrain_params(self, unconstrained_params) -> np.ndarray:
        K = self.n_regimes
        u = np.asarray(unconstrained_params, dtype=float)

        diag         = np.empty(len(u))
        diag[0]      = 1.0 - np.tanh(u[0]) ** 2   # phi
        diag[1:1 + K] = np.exp(u[1:1 + K])          # sigma2_k
        diag[1 + K]  = np.exp(u[1 + K])             # tau2

        log_p  = u[2 + K:].reshape(K, K)
        log_p -= log_p.max(axis=1, keepdims=True)
        p_rows = np.exp(log_p) / np.exp(log_p).sum(axis=1, keepdims=True)
        diag[2 + K:] = (p_rows * (1.0 - p_rows)).ravel()

        return np.diag(diag)
