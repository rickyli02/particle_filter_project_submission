import numpy as np
import pandas as pd

# logsumexp
def logsumexp(a, axis=None):
    a_max = np.max(a, axis=axis, keepdims=True)
    return a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))

# softmax
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

# log_normal_pdf_scalar
def log_normal_pdf_scalar(x, mean, var):
    return -0.5 * np.log(2 * np.pi * var) - 0.5 * ((x - mean) ** 2) / var

