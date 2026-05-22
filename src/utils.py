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


def log_normal_pdf_scalar(y, mean, var):
    """Log density of scalar N(mean, var). Returns -inf for non-positive variance."""
    if var <= 0 or not np.isfinite(var):
        return -np.inf
    return -0.5 * (np.log(2.0 * np.pi * var) + ((y - mean) ** 2) / var)


def log_normal_pdf(y, mean, sd):
    """Log density of N(mean, sd^2), vectorized."""
    return -0.5 * np.log(2.0 * np.pi * sd ** 2) - 0.5 * ((y - mean) / sd) ** 2


def systematic_resample(weights, rng):
    """Systematic resampling with an explicit RNG for reproducible seeds."""
    N = len(weights)
    positions = (rng.random() + np.arange(N)) / N
    cumsum = np.cumsum(weights)
    indices = np.zeros(N, dtype=int)
    i = j = 0
    while i < N:
        if positions[i] < cumsum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    return indices
