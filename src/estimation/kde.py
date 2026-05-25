"""
Kernel Density Estimation
=========================
Weighted KDE for particle filter posteriors and MCMC chain marginals.

Primary use cases
-----------------
  kde = KDE(particles, weights)       # particle filter posterior at one time step
  kde = KDE(chain[:, k])              # PMMH marginal for parameter k (equal weights)

  density   = kde(x_grid)             # evaluate density on a grid
  log_dens  = kde.log_evaluate(x)     # log density (numerically stable)
  samples   = kde.sample(1000)        # draw samples from the KDE

Convenience helpers
-------------------
  particle_posterior_kde(pf, t)       # extract KDE from a ParticleFilter at time t
  chain_marginal_kdes(chain, names)   # list of KDEs, one per parameter column
"""

# Note: do not use KDE within PMMH unless you can verify that the resulting density is unbiased

from __future__ import annotations

import numpy as np


# ── bandwidth selection ───────────────────────────────────────────────────────

def _effective_n(weights: np.ndarray) -> float:
    """Kish effective sample size: 1 / sum(w^2) for normalized weights."""
    w = np.asarray(weights, dtype=float)
    return 1.0 / float(np.sum(w ** 2))


def _weighted_std(particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted standard deviation along each column of particles."""
    p = np.atleast_2d(particles) if particles.ndim == 1 else particles
    w = weights[:, None] if particles.ndim > 1 else weights
    mean = np.sum(w * p if particles.ndim > 1 else weights * particles, axis=0)
    var  = np.sum(w * (p - mean) ** 2 if particles.ndim > 1
                  else weights * (particles - mean) ** 2, axis=0)
    return np.sqrt(var)


def silverman_bandwidth(particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Silverman's rule-of-thumb bandwidth.

    h_k = 1.06 * σ_k * N_eff^{-1/(d+4)}

    where σ_k is the weighted std of dimension k and N_eff is the effective
    sample size.  For 1-D data (d=1) this reduces to the standard
    h = 1.06 * σ * N_eff^{-1/5}.

    Returns a 1-D array of per-dimension bandwidths.
    """
    p = np.atleast_2d(particles) if particles.ndim == 1 else particles
    d      = p.shape[1] if p.ndim == 2 else 1
    n_eff  = _effective_n(weights)
    sigma  = _weighted_std(particles, weights)
    return 1.06 * sigma * n_eff ** (-1.0 / (d + 4.0))


def scott_bandwidth(particles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Scott's rule bandwidth:  h_k = 1.059 * σ_k * N_eff^{-1/(d+4)}.

    Nearly identical to Silverman's; slightly wider.
    """
    p = np.atleast_2d(particles) if particles.ndim == 1 else particles
    d      = p.shape[1] if p.ndim == 2 else 1
    n_eff  = _effective_n(weights)
    sigma  = _weighted_std(particles, weights)
    return 1.059 * sigma * n_eff ** (-1.0 / (d + 4.0))


# ── core KDE class ────────────────────────────────────────────────────────────

class KDE:
    """
    Weighted kernel density estimate for 1-D or multivariate data.

    Parameters
    ----------
    particles : (N,) or (N, d) array-like
        Support points (e.g. particle positions).
    weights : (N,) array-like or None
        Non-negative weights; normalized internally.  None means equal weights.
    bandwidth : float | array-like | 'silverman' | 'scott'
        Scalar or per-dimension bandwidth, or a rule name.
        For 1-D data a float is used directly.
        For multivariate data a float is broadcast across dimensions.
    kernel : 'gaussian' | 'epanechnikov'
        Kernel function.  Gaussian is recommended for smooth posteriors;
        Epanechnikov is MSE-optimal with compact support.
    """

    def __init__(
        self,
        particles,
        weights=None,
        bandwidth: float | str = "silverman",
        kernel: str = "gaussian",
    ):
        particles = np.asarray(particles, dtype=float)
        self._1d = particles.ndim == 1
        self.particles = particles                    # (N,) or (N, d)
        N = particles.shape[0]

        if weights is None:
            self.weights = np.ones(N) / N
        else:
            w = np.asarray(weights, dtype=float)
            if w.shape != (N,):
                raise ValueError(f"weights shape {w.shape} must be ({N},).")
            if np.any(w < 0):
                raise ValueError("weights must be non-negative.")
            self.weights = w / w.sum()

        if kernel not in ("gaussian", "epanechnikov"):
            raise ValueError(f"Unknown kernel '{kernel}'. Choose 'gaussian' or 'epanechnikov'.")
        self.kernel = kernel

        # Resolve bandwidth to a (d,) array or scalar for 1-D
        if isinstance(bandwidth, str):
            if bandwidth == "silverman":
                h = silverman_bandwidth(particles, self.weights)
            elif bandwidth == "scott":
                h = scott_bandwidth(particles, self.weights)
            else:
                raise ValueError(f"Unknown bandwidth rule '{bandwidth}'.")
        else:
            h = np.asarray(bandwidth, dtype=float)
            if h.ndim == 0:
                # scalar → broadcast to all dims
                d = 1 if self._1d else particles.shape[1]
                h = np.full(d, float(h))

        self._h = np.atleast_1d(h)   # (d,) array

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def bandwidth(self) -> np.ndarray:
        """Per-dimension bandwidth array (shape (d,))."""
        return self._h

    @property
    def n_eff(self) -> float:
        """Kish effective sample size."""
        return _effective_n(self.weights)

    def __repr__(self) -> str:
        d = 1 if self._1d else self.particles.shape[1]
        N = self.particles.shape[0]
        return (
            f"KDE(N={N}, d={d}, kernel={self.kernel!r}, "
            f"bandwidth={self._h if d > 1 else float(self._h[0]):.4g}, "
            f"n_eff={self.n_eff:.1f})"
        )

    # ── evaluation ────────────────────────────────────────────────────────────

    def log_evaluate(self, x) -> np.ndarray:
        """
        Log density  log f(x)  evaluated at each point in x.

        Parameters
        ----------
        x : scalar | (M,) | (M, d)

        Returns
        -------
        (M,) array of log densities (or scalar if x is scalar)
        """
        x = np.asarray(x, dtype=float)
        scalar_in = x.ndim == 0
        x = np.atleast_1d(x)

        if self._1d:
            # u : (M, N)
            u = (x[:, None] - self.particles[None, :]) / self._h[0]
            log_k = self._log_kernel_1d(u)                          # (M, N)
            log_dens = (
                np.log(self.weights)[None, :]                       # (1, N)
                + log_k
            )
            result = np.logaddexp.reduce(log_dens, axis=1) - np.log(self._h[0])
        else:
            d = self.particles.shape[1]
            # u : (M, N, d)
            u = (x[:, None, :] - self.particles[None, :, :]) / self._h[None, None, :]
            log_k = self._log_kernel_nd(u)                          # (M, N)
            log_dens = np.log(self.weights)[None, :] + log_k       # (M, N)
            result = (
                np.logaddexp.reduce(log_dens, axis=1)
                - np.sum(np.log(self._h))
            )

        return float(result[0]) if scalar_in else result

    def evaluate(self, x) -> np.ndarray:
        """Density f(x); wraps log_evaluate."""
        return np.exp(self.log_evaluate(x))

    def __call__(self, x) -> np.ndarray:
        return self.evaluate(x)

    # ── sampling ──────────────────────────────────────────────────────────────

    def sample(self, n: int, seed=None) -> np.ndarray:
        """
        Draw n samples from the KDE (smoothed bootstrap).

        Algorithm: resample a particle proportional to its weight, then
        add kernel noise at the chosen bandwidth.

        Returns
        -------
        (n,) for 1-D KDE, (n, d) for multivariate.
        """
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self.weights), size=n, p=self.weights)
        base = self.particles[idx]   # (n,) or (n, d)

        if self.kernel == "gaussian":
            noise = rng.standard_normal(base.shape) * self._h
        else:
            # Epanechnikov: uniform on [-√5, √5] scaled by h gives the
            # correct marginal bandwidth (standard trick)
            noise = rng.uniform(-1.0, 1.0, base.shape) * self._h * np.sqrt(5)

        return base + noise

    # ── kernel helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _log_kernel_1d(u: np.ndarray) -> np.ndarray:
        """Log Gaussian or Epanechnikov kernel values for standardised residuals u."""
        return -0.5 * u ** 2 - 0.5 * np.log(2.0 * np.pi)

    def _log_kernel_nd(self, u: np.ndarray) -> np.ndarray:
        """
        Log multivariate Gaussian kernel for u of shape (M, N, d).

        Uses a diagonal bandwidth (product of 1-D kernels).
        """
        # sum of squared standardised residuals: (M, N)
        log_k = -0.5 * np.sum(u ** 2, axis=2) - 0.5 * u.shape[2] * np.log(2.0 * np.pi)
        return log_k


# ── convenience helpers ───────────────────────────────────────────────────────

def particle_posterior_kde(
    pf,
    t: int,
    state_idx: int = 0,
    bandwidth: float | str = "silverman",
    kernel: str = "gaussian",
) -> KDE:
    """
    Build a KDE from a ParticleFilter's stored history at time step t.

    Parameters
    ----------
    pf        : ParticleFilter  (must have particle_history / weight_history)
    t         : time step index (0-based)
    state_idx : which state component to use (for multivariate latent states)
    bandwidth, kernel : passed to KDE

    Returns
    -------
    KDE over p(x_{state_idx, t} | y_{0:t})
    """
    particles = pf.particle_history[t]
    weights   = pf.weight_history[t].flatten()

    if particles.ndim == 2:
        particles = particles[:, state_idx]
    elif particles.ndim > 2:
        particles = particles[:, state_idx]

    return KDE(particles.flatten(), weights, bandwidth=bandwidth, kernel=kernel)


def chain_marginal_kdes(
    chain: np.ndarray,
    param_names: list[str] | None = None,
    burn_in: int = 0,
    bandwidth: float | str = "silverman",
    kernel: str = "gaussian",
) -> list[KDE]:
    """
    Build one KDE per column of an MCMC chain (e.g. PMMH output).

    Parameters
    ----------
    chain       : (n_iter, d) array of chain samples
    param_names : optional list of d names (for labelling only; stored as kde.name)
    burn_in     : number of initial rows to discard
    bandwidth, kernel : passed to each KDE

    Returns
    -------
    List of d KDE objects; access kde.name for the parameter name.
    """
    samples = chain[burn_in:]
    d = samples.shape[1] if samples.ndim == 2 else 1
    names = param_names or [f"param_{k}" for k in range(d)]

    kdes = []
    for k in range(d):
        col = samples[:, k] if samples.ndim == 2 else samples
        kde = KDE(col, bandwidth=bandwidth, kernel=kernel)
        kde.name = names[k]   # attach name for downstream plotting
        kdes.append(kde)
    return kdes
