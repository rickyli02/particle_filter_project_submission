"""
Regime-switching growth model.

Latent process
--------------
    Regime:  P(s_t = j | s_{t-1} = i) = P_{ij}
    Gap:     x_t = φ x_{t-1} + σ_{s_t} ε_t,   ε_t ~ N(0, 1)

Observation
-----------
    y_t = g*_{s_t} + μ (x_t - x_{t-1}) + τ η_t,   η_t ~ N(0, 1)

where y_t is observed real GDP growth, x_t is the unobserved output gap,
g*_{s_t} is the regime-specific potential growth rate, and μ scales the
contribution of gap changes to growth.

Augmented state  a_t = [x_t, x_{t-1}]'  lifts this to a standard linear
Gaussian SSM conditional on the regime path:

    a_t = F a_{t-1} + ε_t,        ε_t ~ N(0, Q_{s_t})
    y_t = H a_t + d_{s_t} + η_t,  η_t ~ N(0, R)

    F      = [[φ, 0],             (shared across regimes)
              [1, 0]]

    Q_j    = [[σ_j², 0],          (regime-specific)
              [0,    0]]

    H      = [[μ, −μ]],           (shared; 1 × 2)

    d_j    = [g*_j],              (regime-specific intercept)

    R      = [[τ²]]               (shared; 1 × 1)

Parameters
----------
θ = [p11, p22, φ, σ1, σ2, g1, g2, μ, τ]   (9 constrained params)
z = [logit p11, logit p22, arctanh φ,       (9 unconstrained params)
     log σ1, log σ2, g1, g2, μ, log τ]

Regime s=0 is the expansion regime (low volatility, higher g*).
Regime s=1 is the contraction regime (high volatility, lower g*).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import multivariate_normal

from models.regime_switching_base import (
    RegimeSwitchingBase,
    RegimeSwitchingDims,
    RegimeSwitchingStructure,
)
from utils import symmetrize


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-float(x)))


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return np.log(p / (1.0 - p))


def _build_matrices(phi: float, sigma1: float, sigma2: float,
                    g1: float, g2: float, mu: float, tau: float):
    """
    Construct per-regime and shared SSM matrices from scalar parameters.

    Returns F, Q_list, H, d_list, R.
    """
    F = np.array([[phi, 0.0],
                  [1.0, 0.0]], dtype=float)

    Q_list = [
        np.array([[sigma1 ** 2, 0.0],
                  [0.0,         0.0]], dtype=float),
        np.array([[sigma2 ** 2, 0.0],
                  [0.0,         0.0]], dtype=float),
    ]

    H = np.array([[mu, -mu]], dtype=float)   # shape (1, 2)

    d_list = [
        np.array([g1], dtype=float),
        np.array([g2], dtype=float),
    ]

    R = np.array([[tau ** 2]], dtype=float)  # shape (1, 1)

    return F, Q_list, H, d_list, R


_DIMS = RegimeSwitchingDims(n_regimes=2, state_dim=2, obs_dim=1)
_STRUCT = RegimeSwitchingStructure(
    regime_specific_A=False,   # F shared
    regime_specific_C=False,   # H shared
    regime_specific_Q=True,    # Q_j per regime
    regime_specific_R=False,   # R shared
    has_state_intercept=False,
    has_obs_intercept=True,    # d_j = [g*_j] per regime
)

_PARAM_NAMES = ["p11", "p22", "phi", "sigma1", "sigma2", "g1", "g2", "mu", "tau"]


class RegimeSwitchingGrowth(RegimeSwitchingBase):
    """
    Regime-switching output gap / GDP growth model.

    Parameters
    ----------
    p11    : float in (0, 1) — probability of staying in expansion
    p22    : float in (0, 1) — probability of staying in contraction
    phi    : float in (-1, 1) — AR(1) persistence of output gap
    sigma1 : float > 0 — output gap shock std in expansion
    sigma2 : float > 0 — output gap shock std in contraction
    g1     : float — potential growth rate in expansion
    g2     : float — potential growth rate in contraction
    mu     : float — slope relating gap changes to observed growth
    tau    : float > 0 — observation noise std
    seed   : int | None
    """

    def __init__(
        self,
        p11: float,
        p22: float,
        phi: float,
        sigma1: float,
        sigma2: float,
        g1: float,
        g2: float,
        mu: float,
        tau: float,
        seed=None,
    ):
        # Set scalars before super().__init__ because check_params_validity is
        # called during the base __init__ and reads these attributes.
        self.p11    = float(p11)
        self.p22    = float(p22)
        self.phi    = float(phi)
        self.sigma1 = float(sigma1)
        self.sigma2 = float(sigma2)
        self.g1     = float(g1)
        self.g2     = float(g2)
        self.mu     = float(mu)
        self.tau    = float(tau)

        P_trans = np.array([[p11,       1.0 - p11],
                            [1.0 - p22, p22      ]], dtype=float)
        F, Q_list, H, d_list, R = _build_matrices(phi, sigma1, sigma2, g1, g2, mu, tau)

        super().__init__(
            dims=_DIMS,
            structure=_STRUCT,
            A=F,
            C=H,
            Q=Q_list,
            R=R,
            P_trans=P_trans,
            d=d_list,
            seed=seed,
        )
        # Override the matrix-keyed params_dict from the base with scalar params
        self.params_dict = {
            "p11":    p11,
            "p22":    p22,
            "phi":    phi,
            "sigma1": sigma1,
            "sigma2": sigma2,
            "g1":     g1,
            "g2":     g2,
            "mu":     mu,
            "tau":    tau,
        }

    # ── repr / describe ───────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"RegimeSwitchingGrowth("
            f"p11={self.p11!r}, p22={self.p22!r}, phi={self.phi!r}, "
            f"sigma1={self.sigma1!r}, sigma2={self.sigma2!r}, "
            f"g1={self.g1!r}, g2={self.g2!r}, mu={self.mu!r}, tau={self.tau!r})"
        )

    def describe(self) -> str:
        pi = self.regime_probabilities_stationary
        return (
            f"RegimeSwitchingGrowth\n"
            f"  y_t = g*_{{s_t}} + mu*(x_t - x_{{t-1}}) + tau*eta_t\n"
            f"  x_t = phi*x_{{t-1}} + sigma_{{s_t}}*eps_t\n"
            f"  Regime 0 (expansion): g*={self.g1:.4f}, sigma={self.sigma1:.4f}, "
            f"stationary prob={pi[0]:.4f}\n"
            f"  Regime 1 (contraction): g*={self.g2:.4f}, sigma={self.sigma2:.4f}, "
            f"stationary prob={pi[1]:.4f}\n"
            f"  phi={self.phi:.4f},  mu={self.mu:.4f},  tau={self.tau:.4f}\n"
            f"  p11={self.p11:.4f},  p22={self.p22:.4f}"
        )

    # ── validity ──────────────────────────────────────────────────────────────

    def check_params_validity(self):
        if not (-1.0 < self.phi < 1.0):
            raise ValueError(f"phi={self.phi}: output gap must be stationary (|phi| < 1).")
        if self.sigma1 <= 0.0 or self.sigma2 <= 0.0:
            raise ValueError("sigma1 and sigma2 must be positive.")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive.")
        if not (0.0 < self.p11 < 1.0 and 0.0 < self.p22 < 1.0):
            raise ValueError("p11 and p22 must be in (0, 1).")
        # Delegate matrix checks to the base
        super().check_params_validity()

    # ── initial distribution ──────────────────────────────────────────────────

    def _mixture_cov(self) -> np.ndarray:
        """
        Stationary covariance of a_t = [x_t, x_{t-1}] marginalised over regimes.

        For F = [[phi, 0], [1, 0]], Q_avg = diag(avg_sigma^2, 0):
            P[0,0] = P[1,1] = avg_sigma^2 / (1 - phi^2)
            P[0,1] = P[1,0] = phi * P[0,0]
        """
        pi        = self.regime_probabilities_stationary
        avg_sigma2 = pi[0] * self.sigma1 ** 2 + pi[1] * self.sigma2 ** 2
        if abs(self.phi) < 1.0:
            var_x = avg_sigma2 / (1.0 - self.phi ** 2)
        else:
            var_x = 10.0 * avg_sigma2
        cov_lag = self.phi * var_x
        return np.array([[var_x,   cov_lag],
                         [cov_lag, var_x  ]], dtype=float)

    def sample_initial_distribution(self):
        """
        Sample (a_0, s_0) where a_0 = [x_0, x_{-1}]' from the stationary mixture.

        Returns (a_0, s_0).
        """
        pi = self.regime_probabilities_stationary
        s0 = int(self.rng.choice(2, p=pi))
        P0 = self._mixture_cov()
        a0 = self.rng.multivariate_normal(np.zeros(2), P0)
        return a0, s0

    def initial_density(self, a, regime: int) -> float:
        """p(a_0 = a, s_0 = regime) = π_{regime} · N(a; 0, P_mix)."""
        pi = float(self.regime_probabilities_stationary[regime])
        P0 = self._mixture_cov()
        return pi * float(multivariate_normal.pdf(a, mean=np.zeros(2), cov=P0))

    # ── update_params ─────────────────────────────────────────────────────────

    def update_params(self, constrained_params):
        """
        Rebuild all model matrices from a 9-element constrained parameter list.

        Parameters
        ----------
        constrained_params : array-like of length 9
            [p11, p22, phi, sigma1, sigma2, g1, g2, mu, tau]
        """
        p11, p22, phi, sigma1, sigma2, g1, g2, mu, tau = constrained_params

        self.p11    = float(p11)
        self.p22    = float(p22)
        self.phi    = float(phi)
        self.sigma1 = float(sigma1)
        self.sigma2 = float(sigma2)
        self.g1     = float(g1)
        self.g2     = float(g2)
        self.mu     = float(mu)
        self.tau    = float(tau)

        self.regime_transition_matrix = np.array(
            [[p11,       1.0 - p11],
             [1.0 - p22, p22      ]], dtype=float
        )
        self.regime_probabilities_stationary = self.solve_stationary_distribution()

        F, Q_list, H, d_list, R = _build_matrices(phi, sigma1, sigma2, g1, g2, mu, tau)
        self.A_list = [F, F]
        self.C_list = [H, H]
        self.Q_list = Q_list
        self.R_list = [R, R]
        self.d_list = d_list

        self.params_dict = {
            "p11":    p11,    "p22":   p22,
            "phi":    phi,    "sigma1": sigma1,
            "sigma2": sigma2, "g1":    g1,
            "g2":     g2,     "mu":    mu,
            "tau":    tau,
        }
        self.check_params_validity()

    # ── parameter transforms (for MCMC / MLE) ────────────────────────────────

    def unconstrain_params(self, constrained_params) -> np.ndarray:
        """
        Map constrained θ = [p11, p22, phi, sigma1, sigma2, g1, g2, mu, tau]
        to unconstrained z = [logit p11, logit p22, arctanh phi,
                               log sigma1, log sigma2, g1, g2, mu, log tau].
        """
        p11, p22, phi, sigma1, sigma2, g1, g2, mu, tau = constrained_params
        return np.array([
            _logit(p11),
            _logit(p22),
            np.arctanh(phi),
            np.log(sigma1),
            np.log(sigma2),
            g1,
            g2,
            mu,
            np.log(tau),
        ], dtype=float)

    def constrain_params(self, unconstrained_params) -> list:
        """
        Map unconstrained z back to constrained θ.
        Returns a plain list compatible with update_params.
        """
        z = np.asarray(unconstrained_params, dtype=float)
        return [
            _sigmoid(z[0]),   # p11
            _sigmoid(z[1]),   # p22
            float(np.tanh(z[2])),   # phi
            float(np.exp(z[3])),    # sigma1
            float(np.exp(z[4])),    # sigma2
            float(z[5]),            # g1
            float(z[6]),            # g2
            float(z[7]),            # mu
            float(np.exp(z[8])),    # tau
        ]

    def jacobian_constrain_params(self, unconstrained_params) -> np.ndarray:
        """
        Diagonal Jacobian dθ/dz at unconstrained_params.

        Returns a (9, 9) diagonal matrix.  Used by MCMCBase for the
        change-of-variables log-determinant term.
        """
        z = np.asarray(unconstrained_params, dtype=float)
        p11    = _sigmoid(z[0])
        p22    = _sigmoid(z[1])
        phi    = float(np.tanh(z[2]))
        sigma1 = float(np.exp(z[3]))
        sigma2 = float(np.exp(z[4]))
        tau    = float(np.exp(z[8]))
        return np.diag([
            p11 * (1.0 - p11),    # d(sigmoid)/dz
            p22 * (1.0 - p22),
            1.0 - phi ** 2,       # d(tanh)/dz
            sigma1,               # d(exp)/dz
            sigma2,
            1.0,                  # g1, g2, mu unconstrained
            1.0,
            1.0,
            tau,
        ])
