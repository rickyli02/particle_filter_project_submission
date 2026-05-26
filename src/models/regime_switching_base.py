# Base regime-switching model for regime_switching.py, regime_switching_macro.py, and regime_switching_growth.property
# Similar to regime_switching.py
# Used for RBPF, RBPF-PMMH, Kim smoother, and Kim smoother-PMMH

# Has closed form log likelihood conditional on regime path
# (since regime_switching_complex transition probabilities are dependent on latent states, this is not true for that model)

# Class for regime-switching model dimensions
# Class for regime-switching model parameters, specifying which parameters are shared and which are regime-specific
# Regime-switching model base class inherits from base.py StateSpaceModel
# Model takes in a ModelDims and a ModelParams object in initialization
# update_params() respects the ModelParams object

# We will later update regime_switching.py, regime_switching_macro.py, and regime_switching_growth.py to inherit from this class
"""
Base class for Markov-switching linear Gaussian state-space models.

All models in this family share the structure

    P(s_t = j | s_{t-1} = i) = P_{ij}                  (Markov regime chain)

    x_t = A_{s_t} x_{t-1} + b_{s_t} + ε_t,  ε_t ~ N(0, Q_{s_t})
    y_t = C_{s_t} x_t     + d_{s_t} + η_t,  η_t ~ N(0, R_{s_t})

Matrices that are shared across regimes are stored as a single array;
per-regime matrices are stored as lists of length n_regimes.
The RegimeSwitchingStructure dataclass controls which matrices vary.

This class exposes the interface required by:
    KimFilter  — estimation/kim_filter.py
    RBPF       — estimation/rbpf.py  (Rao-Blackwellized PF)

Key method beyond the StateSpaceModel interface:
    log_likelihood_given_regimes(y, regime_path)
        Exact log p(y_{0:T-1} | s_{0:T-1}) via Kalman filter.
        Used by RBPF incremental weights and as a correctness reference.

Subclasses override constrain_params / unconstrain_params /
jacobian_constrain_params to expose domain-specific parameterisations for
MCMC and MLE.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_discrete_lyapunov
from scipy.stats import multivariate_normal

from models.base import StateSpaceModel
from utils import symmetrize


# ── dimension and structure descriptors ──────────────────────────────────────

@dataclass
class RegimeSwitchingDims:
    """Dimensions of a Markov-switching linear Gaussian SSM."""
    n_regimes: int
    state_dim: int
    obs_dim:   int


@dataclass
class RegimeSwitchingStructure:
    """
    Controls which SSM matrices vary by regime.

    A flag of True means each regime has its own matrix.
    A flag of False means a single matrix is shared across all regimes.
    """
    regime_specific_A: bool = True    # state transition matrix
    regime_specific_C: bool = False   # observation matrix
    regime_specific_Q: bool = True    # state noise covariance
    regime_specific_R: bool = False   # observation noise covariance
    has_state_intercept: bool = False  # b_{s_t} in x_t = A x_{t-1} + b + ε
    has_obs_intercept:   bool = False  # d_{s_t} in y_t = C x_t + d + η


# ── base class ────────────────────────────────────────────────────────────────

class RegimeSwitchingBase(StateSpaceModel):
    """
    Base class for Markov-switching linear Gaussian state-space models.

    Parameters
    ----------
    dims      : RegimeSwitchingDims
    structure : RegimeSwitchingStructure
    A         : (n, n) or list of K (n, n) arrays — state transition matrix/matrices
    C         : (m, n) or list of K (m, n) arrays — observation matrix/matrices
    Q         : (n, n) PSD or list of K — state noise covariances
    R         : (m, m) PD  or list of K — observation noise covariances
    P_trans   : (K, K) row-stochastic regime transition matrix
    b         : (n,) or list of K (n,) arrays — state intercepts (optional)
    d         : (m,) or list of K (m,) arrays — obs intercepts (optional)
    seed      : int | None
    """

    def __init__(
        self,
        dims: RegimeSwitchingDims,
        structure: RegimeSwitchingStructure,
        A,
        C,
        Q,
        R,
        P_trans: np.ndarray,
        b=None,
        d=None,
        seed=None,
    ):
        super().__init__(seed=seed, state_dim=dims.state_dim, obs_dim=dims.obs_dim)
        self.dims      = dims
        self.structure = structure
        self.n_regimes = dims.n_regimes
        self.rng       = np.random.default_rng(seed)

        K = self.n_regimes
        self.regime_transition_matrix        = np.asarray(P_trans, dtype=float)
        self.regime_probabilities_stationary = self.solve_stationary_distribution()

        self.A_list = self._to_list(A, K, "A")
        self.C_list = self._to_list(C, K, "C")
        self.Q_list = self._to_list(Q, K, "Q")
        self.R_list = self._to_list(R, K, "R")

        self.b_list: Optional[List[np.ndarray]] = None
        self.d_list: Optional[List[np.ndarray]] = None

        if structure.has_state_intercept:
            self.b_list = (
                self._to_list(b, K, "b")
                if b is not None
                else [np.zeros(dims.state_dim) for _ in range(K)]
            )
        if structure.has_obs_intercept:
            self.d_list = (
                self._to_list(d, K, "d")
                if d is not None
                else [np.zeros(dims.obs_dim) for _ in range(K)]
            )

        self.params_dict = self._build_params_dict()
        self.check_params_validity()

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _to_list(mat, K: int, name: str) -> List[np.ndarray]:
        """Normalise a single array or a list of arrays to a list of length K."""
        if isinstance(mat, (list, tuple)):
            if len(mat) != K:
                raise ValueError(
                    f"'{name}' list must have length n_regimes={K}; got {len(mat)}."
                )
            return [np.asarray(m, dtype=float) for m in mat]
        arr = np.asarray(mat, dtype=float)
        return [arr.copy() for _ in range(K)]

    def _build_params_dict(self) -> dict:
        s = self.structure
        d: dict = {"regime_transition_matrix": self.regime_transition_matrix}
        d["A"] = self.A_list if s.regime_specific_A else self.A_list[0]
        d["C"] = self.C_list if s.regime_specific_C else self.C_list[0]
        d["Q"] = self.Q_list if s.regime_specific_Q else self.Q_list[0]
        d["R"] = self.R_list if s.regime_specific_R else self.R_list[0]
        if s.has_state_intercept:
            d["b"] = self.b_list
        if s.has_obs_intercept:
            d["d"] = self.d_list
        return d

    # Per-regime matrix accessors (always valid regardless of structure flags)
    def _A(self, k: int) -> np.ndarray: return self.A_list[k]
    def _C(self, k: int) -> np.ndarray: return self.C_list[k]
    def _Q(self, k: int) -> np.ndarray: return self.Q_list[k]
    def _R(self, k: int) -> np.ndarray: return self.R_list[k]

    def _b(self, k: int) -> np.ndarray:
        return self.b_list[k] if self.b_list is not None else np.zeros(self.state_dim)

    def _d(self, k: int) -> np.ndarray:
        return self.d_list[k] if self.d_list is not None else np.zeros(self.obs_dim)

    def _stationary_cov(self, k: int) -> np.ndarray:
        """Stationary state covariance for regime k (diffuse fallback for non-stationary A)."""
        A_k = self._A(k)
        Q_k = self._Q(k)
        if np.all(np.abs(np.linalg.eigvals(A_k)) < 1.0):
            return solve_discrete_lyapunov(A_k, Q_k)
        return np.eye(self.state_dim) * (np.trace(Q_k) + 1.0)

    # ── repr / describe ───────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_regimes={self.n_regimes}, "
            f"state_dim={self.state_dim}, obs_dim={self.obs_dim})"
        )

    def describe(self) -> str:
        s  = self.structure
        pr = lambda flag: "per-regime" if flag else "shared"
        lines = [
            f"{self.__class__.__name__}",
            f"  Markov-switching linear Gaussian SSM",
            f"  Regimes: {self.n_regimes},  state_dim: {self.state_dim},  obs_dim: {self.obs_dim}",
            f"  A: {pr(s.regime_specific_A)},  C: {pr(s.regime_specific_C)},  "
            f"  Q: {pr(s.regime_specific_Q)},  R: {pr(s.regime_specific_R)}",
            f"  State intercept: {s.has_state_intercept},  Obs intercept: {s.has_obs_intercept}",
            f"  Stationary regime probs: {np.round(self.regime_probabilities_stationary, 4)}",
            f"  Regime transition matrix:\n{self.regime_transition_matrix}",
        ]
        return "\n".join(lines)

    # ── validity / stationary distribution ───────────────────────────────────

    def check_params_validity(self):
        P = self.regime_transition_matrix
        K = self.n_regimes
        if P.shape != (K, K):
            raise ValueError(
                f"regime_transition_matrix must be ({K}, {K}); got {P.shape}."
            )
        if np.any(P < 0):
            raise ValueError("regime_transition_matrix has negative entries.")
        if not np.allclose(P.sum(axis=1), 1.0):
            raise ValueError("regime_transition_matrix rows must sum to 1.")
        for k in range(K):
            Q_k = self._Q(k)
            R_k = self._R(k)
            if np.any(np.linalg.eigvalsh(Q_k) < -1e-10):
                raise ValueError(f"Q_list[{k}] is not positive semi-definite.")
            try:
                np.linalg.cholesky(R_k)
            except np.linalg.LinAlgError:
                raise ValueError(f"R_list[{k}] is not positive definite.")

    def solve_stationary_distribution(self) -> np.ndarray:
        """Solve the stationary distribution π of the Markov regime chain."""
        K     = self.n_regimes
        P     = self.regime_transition_matrix
        A_mat = P.T - np.eye(K)
        A_mat[-1, :] = 1.0
        b_vec = np.zeros(K)
        b_vec[-1] = 1.0
        return np.linalg.solve(A_mat, b_vec)

    # ── StateSpaceModel interface ─────────────────────────────────────────────

    def sample_initial_distribution(self):
        """
        Sample (x_0, s_0) from the joint stationary distribution.

        Returns
        -------
        x_0 : (state_dim,) float array
        s_0 : int — initial regime index
        """
        pi = self.regime_probabilities_stationary
        s0 = int(self.rng.choice(self.n_regimes, p=pi))
        P0 = self._stationary_cov(s0)
        x0 = self.rng.multivariate_normal(np.zeros(self.state_dim), P0)
        return x0, s0

    def initial_density(self, x, regime: int) -> float:
        """
        Joint density p(x_0 = x, s_0 = regime).

        = π_{regime} · N(x ; 0, P_stationary_{regime})
        """
        pi = float(self.regime_probabilities_stationary[regime])
        P0 = self._stationary_cov(regime)
        return pi * float(multivariate_normal.pdf(x, mean=np.zeros(self.state_dim), cov=P0))

    def transition(self, x_prev, regime: int) -> np.ndarray:
        """Sample x_t | x_{t-1} = x_prev, s_t = regime."""
        A_k   = self._A(regime)
        Q_k   = self._Q(regime)
        b_k   = self._b(regime)
        noise = self.rng.multivariate_normal(np.zeros(self.state_dim), Q_k)
        return A_k @ np.asarray(x_prev, dtype=float) + b_k + noise

    def observation(self, x, regime: int) -> np.ndarray:
        """Sample y_t | x_t = x, s_t = regime."""
        C_k   = self._C(regime)
        R_k   = self._R(regime)
        d_k   = self._d(regime)
        noise = self.rng.multivariate_normal(np.zeros(self.obs_dim), R_k)
        return C_k @ np.asarray(x, dtype=float) + d_k + noise

    def log_transition_density(self, x_next, x_prev, regime: int) -> float:
        """log p(x_t = x_next | x_{t-1} = x_prev, s_t = regime)."""
        A_k  = self._A(regime)
        Q_k  = self._Q(regime)
        b_k  = self._b(regime)
        mean = A_k @ np.asarray(x_prev, dtype=float) + b_k
        return float(multivariate_normal.logpdf(x_next, mean=mean, cov=Q_k))

    def log_observation_density(self, y, x, regime: int) -> float:
        """log p(y_t | x_t = x, s_t = regime)."""
        C_k  = self._C(regime)
        R_k  = self._R(regime)
        d_k  = self._d(regime)
        mean = C_k @ np.asarray(x, dtype=float) + d_k
        return float(multivariate_normal.logpdf(y, mean=mean, cov=R_k))

    def generate_data(self, num_time_steps: int):
        """
        Simulate data from the regime-switching model.

        Returns
        -------
        states  : (T, state_dim) float array
        regimes : (T,) int array of regime indices
        obs     : (T, obs_dim) float array
        """
        T   = num_time_steps
        P   = self.regime_transition_matrix
        states  = np.zeros((T, self.state_dim))
        regimes = np.zeros(T, dtype=int)
        obs     = np.zeros((T, self.obs_dim))

        x0, s0     = self.sample_initial_distribution()
        states[0]  = x0
        regimes[0] = s0
        obs[0]     = self.observation(x0, s0)

        for t in range(1, T):
            s_prev     = regimes[t - 1]
            s_t        = int(self.rng.choice(self.n_regimes, p=P[s_prev]))
            regimes[t] = s_t
            states[t]  = self.transition(states[t - 1], s_t)
            obs[t]     = self.observation(states[t], s_t)

        return states, regimes, obs

    # ── closed-form conditional log-likelihood ────────────────────────────────

    def log_likelihood_given_regimes(
        self,
        y: np.ndarray,
        regime_path: np.ndarray,
    ) -> float:
        """
        Exact log p(y_{0:T-1} | s_{0:T-1}) via Kalman filter.

        Conditioned on a known regime sequence the model is a time-varying
        linear Gaussian SSM, so the Kalman filter delivers the exact marginal
        likelihood with no approximation.

        Parameters
        ----------
        y           : (T,) or (T, obs_dim) observations
        regime_path : (T,) int array of regime indices in 0..K-1

        Returns
        -------
        loglik : float — exact log p(y | s_{0:T-1}); -inf if numerically invalid
        """
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y[:, None]
        T, m   = y.shape
        n      = self.state_dim
        eye_n  = np.eye(n)
        loglik = 0.0

        # Initial prior: stationary covariance for s_0
        s0    = int(regime_path[0])
        mu    = np.zeros(n)
        P_cov = symmetrize(self._stationary_cov(s0))

        for t in range(T):
            s_t = int(regime_path[t])
            A_t = self._A(s_t)
            C_t = self._C(s_t)
            Q_t = self._Q(s_t)
            R_t = self._R(s_t)
            b_t = self._b(s_t)
            d_t = self._d(s_t)
            y_t = y[t]

            # Predict
            mu_pred = A_t @ mu + b_t
            P_pred  = symmetrize(A_t @ P_cov @ A_t.T + Q_t)

            # Innovation
            v = y_t - C_t @ mu_pred - d_t
            S = symmetrize(C_t @ P_pred @ C_t.T + R_t)

            try:
                cf = cho_factor(S)
            except np.linalg.LinAlgError:
                return -np.inf

            log_det_S = 2.0 * np.sum(np.log(np.diag(cf[0])))
            loglik   -= 0.5 * (m * np.log(2.0 * np.pi) + log_det_S + v @ cho_solve(cf, v))

            # Update (Joseph form for numerical stability)
            K_gain = cho_solve(cf, C_t @ P_pred).T   # (n, m)
            I_KC   = eye_n - K_gain @ C_t
            mu     = mu_pred + K_gain @ v
            P_cov  = symmetrize(I_KC @ P_pred @ I_KC.T + K_gain @ R_t @ K_gain.T)

        return float(loglik)

    # ── update_params ─────────────────────────────────────────────────────────

    def update_params(self, constrained_params: dict):
        """
        Update model matrices in-place from a dict of constrained parameters.

        Recognised keys (all optional; only keys present are updated):
            'regime_transition_matrix'  : (K, K) row-stochastic array
            'A'  : single matrix (if shared) or list of K matrices
            'C'  : single matrix (if shared) or list of K matrices
            'Q'  : single matrix (if shared) or list of K matrices
            'R'  : single matrix (if shared) or list of K matrices
            'b'  : single vector or list of K vectors (if has_state_intercept)
            'd'  : single vector or list of K vectors (if has_obs_intercept)
        """
        K = self.n_regimes
        p = constrained_params

        if "regime_transition_matrix" in p:
            self.regime_transition_matrix = np.asarray(
                p["regime_transition_matrix"], dtype=float
            )
            self.regime_probabilities_stationary = self.solve_stationary_distribution()

        for key, attr in (("A", "A_list"), ("C", "C_list"),
                           ("Q", "Q_list"), ("R", "R_list")):
            if key in p:
                setattr(self, attr, self._to_list(p[key], K, key))

        if "b" in p and self.structure.has_state_intercept:
            self.b_list = self._to_list(p["b"], K, "b")
        if "d" in p and self.structure.has_obs_intercept:
            self.d_list = self._to_list(p["d"], K, "d")

        self.params_dict = self._build_params_dict()
        self.check_params_validity()