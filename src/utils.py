import numpy as np


def logsumexp(a, axis=None):
    """Numerically stable log(sum(exp(a)))."""
    a = np.asarray(a, dtype=float)
    if axis is None:
        a_max = float(np.max(a))
        return a_max + np.log(np.sum(np.exp(a - a_max)))
    a_max = np.max(a, axis=axis, keepdims=True)
    return a_max.squeeze(axis=axis) + np.log(np.sum(np.exp(a - a_max), axis=axis))


def softmax(x):
    """Row-wise softmax."""
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def row_softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax (each row sums to 1)."""
    x_s = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x_s)
    return e / np.sum(e, axis=1, keepdims=True)


def symmetrize(m: np.ndarray) -> np.ndarray:
    """Return (m + m.T) / 2 to enforce symmetry of a covariance matrix."""
    return 0.5 * (m + m.T)


def log_normal_pdf_scalar(y, mean, var):
    """Log density of scalar N(mean, var). Returns -inf for non-positive variance."""
    if var <= 0 or not np.isfinite(var):
        return -np.inf
    return -0.5 * (np.log(2.0 * np.pi * var) + ((y - mean) ** 2) / var)


def log_normal_pdf(y, mean, sd):
    """Log density of N(mean, sd^2), vectorized."""
    return -0.5 * np.log(2.0 * np.pi * sd ** 2) - 0.5 * ((y - mean) / sd) ** 2


# ── particle filter diagnostics ───────────────────────────────────────────────

def filtered_trajectory(pf, state_idx=0):
    """Weighted posterior mean of state component `state_idx` at each time step."""
    out = []
    for particles, weights in zip(pf.particle_history, pf.weight_history):
        w = weights.flatten()
        vals = particles if particles.ndim == 1 else particles[:, state_idx]
        out.append(np.average(vals, weights=w))
    return np.array(out)


def ess_trajectory(pf):
    """Effective sample size 1/Σw² at each time step."""
    return np.array([1.0 / np.sum(w.flatten() ** 2) for w in pf.weight_history])


def rmse(true, est):
    """Root mean squared error between two array-like sequences."""
    return np.sqrt(np.mean((np.asarray(true) - np.asarray(est)) ** 2))

def _logsumexp2d(a: np.ndarray) -> float:
    """Numerically stable logsumexp over all elements of a 2-D array."""
    a_max = a.max()
    return float(a_max + np.log(np.exp(a - a_max).sum()))