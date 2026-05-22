import numpy as np
from scipy.stats import norm
import pandas as pd
from tqdm import tqdm

# 1. 
# Generate synthetic data from a linear state-space model
# Takes in the number of time steps, the state transition matrix, the observation matrix, and the noise parameters
def generate_synthetic_data(num_time_steps, A, C, Q, R):
    # verify input dimensions
    assert A.shape[0] == A.shape[1], "A must be a square matrix"
    assert C.shape[1] == A.shape[0], "C must have the same number of columns as A has rows"
    assert Q.shape == (A.shape[0], A.shape[0]), "Q must be a square matrix with the same dimensions as A"
    assert R.shape == (C.shape[0], C.shape[0]), "R must be a square matrix with the same dimensions as C @ A @ C.T"

    # verify that resulting time series has weak stationarity properties (e.g., eigenvalues of A should be less than 1 in magnitude)
    eigenvalues = np.linalg.eigvals(A)
    if np.any(np.abs(eigenvalues) >= 1):
        raise ValueError("The state transition matrix A must have eigenvalues with magnitude less than 1 for the process to be stationary.")

    # Initialize the state and observation arrays
    states = np.zeros((num_time_steps, A.shape[0]))
    observations = np.zeros((num_time_steps, C.shape[0]))

    # Generate the initial state from stationary distribution (assuming A is stable)
    states[0] = norm.rvs(size=A.shape[0])

    # Generate the states and observations over time
    for t in tqdm(range(1, num_time_steps)):
        # State transition
        states[t] = A @ states[t-1] + norm.rvs(scale=np.sqrt(Q), size=A.shape[0])
        # Observation
        observations[t] = C @ states[t] + norm.rvs(scale=np.sqrt(R), size=C.shape[0])

    return states, observations

# 2.
# Generate synthetic data from a regime-switching model, with linear state-space models for each regime
def generate_regime_switching_data(num_time_steps, A_list, C_list, Q_list, R_list, regime_transition_matrix, regime_probabilities):
    num_regimes = len(A_list)

    # verify input dimensions
    assert len(C_list) == num_regimes, "Length of C_list must match number of regimes"
    assert len(Q_list) == num_regimes, "Length of Q_list must match number of regimes"
    assert len(R_list) == num_regimes, "Length of R_list must match number of regimes"

    states = np.zeros((num_time_steps, A_list[0].shape[0]))
    observations = np.zeros((num_time_steps, C_list[0].shape[0]))
    regimes = np.zeros(num_time_steps, dtype=int)

    # Generate the initial state and regime
    states[0] = norm.rvs(size=A_list[0].shape[0])
    regimes[0] = np.random.choice(num_regimes, p=regime_probabilities)

    for t in tqdm(range(1, num_time_steps)):
        # Sample the regime for the current time step
        regimes[t] = np.random.choice(num_regimes, p=regime_probabilities)

        # Get the parameters for the current regime
        A = A_list[regimes[t]]
        C = C_list[regimes[t]]
        Q = Q_list[regimes[t]]
        R = R_list[regimes[t]]

        # State transition
        states[t] = A @ states[t-1] + norm.rvs(scale=np.sqrt(Q), size=A.shape[0])
        # Observation
        observations[t] = C @ states[t] + norm.rvs(scale=np.sqrt(R), size=C.shape[0])

    return states, observations, regimes