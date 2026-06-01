"""
Rao-Blackwellized Particle Filter for Markov-switching linear Gaussian SSMs.

Particles track only the discrete regime sequence s_t.  The continuous latent
state a_t is marginalised analytically via a per-particle Kalman filter,
exploiting the conditional linear Gaussian structure.

Variance reduction vs. the bootstrap PF: the d-dimensional continuous state
is integrated out exactly, so only the K-state Markov chain is sampled.

Compatible with any model that exposes:
    n_regimes, regime_transition_matrix, regime_probabilities_stationary
    A_list, C_list, Q_list, R_list        — per-regime SSM matrices
    state_dim, obs_dim
    b_list (optional)                     — per-regime state intercepts
    d_list (optional)                     — per-regime obs intercepts

RegimeSwitchingBase (models/regime_switching_base.py) satisfies this interface.
"""

from __future__ import annotations

import time
import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from estimation.particle_filter import ParticleFilter
from utils import logsumexp


class RaoBlackwellizedParticleFilter(ParticleFilter):
    """
    Rao-Blackwellized Particle Filter.

    Each particle is a regime index s_t^(n).  Associated with each particle is
    a Kalman filter (mu^(n), P^(n)) that tracks the distribution of the
    continuous state conditioned on the particle's regime path.  Particle
    weights are the Kalman predictive log-likelihoods.

    Parameters
    ----------
    model           : regime-switching SSM (must expose the RBPF interface)
    N_particles     : int — number of regime particles
    data            : (T,) or (T, obs_dim) observations
    resample_method : ResamplingMethod | None — resamples when ESS < threshold * N
    seed            : int | None
    """

    _REQUIRED = (
        "n_regimes",
        "regime_transition_matrix",
        "regime_probabilities_stationary",
        "A_list", "C_list", "Q_list", "R_list",
    )

    def __init__(
        self,
        model=None,
        N_particles: int = 1000,
        data=None,
        resample_method=None,
        seed=None,
    ):
        super().__init__(model, N_particles, data, resample_method, seed)
        self.check_model()
        # Set by run_filter
        self.regime_prob_history: np.ndarray | None = None

    def __repr__(self) -> str:
        K = getattr(self.model, "n_regimes", "?")
        return (
            f"RaoBlackwellizedParticleFilter("
            f"model={self.model!r}, N_particles={self.N_particles}, n_regimes={K})"
        )

    # ── model validation ──────────────────────────────────────────────────────

    def check_model(self):
        missing = [a for a in self._REQUIRED if not hasattr(self.model, a)]
        if missing:
            raise ValueError(
                f"RBPF requires model attributes {missing}; "
                f"{type(self.model).__name__} does not provide them."
            )

    # ── filter ────────────────────────────────────────────────────────────────

    def _kalman_update_step(self, y_t, s_particles, mu_p, P_p, ssm_lists):
        """
        Vectorised Kalman predict/update for all regime particles at one timestep.

        For each regime k, runs a batched Kalman filter over the Nk particles
        currently in that regime and returns their log-weights and updated
        (mean, covariance) pairs.

        Parameters
        ----------
        y_t        : (m,) observation at time t
        s_particles: (N,) int  current regime index per particle
        mu_p       : (N, n)    prior Kalman means
        P_p        : (N, n, n) prior Kalman covariances
        ssm_lists  : dict with keys A_list, C_list, Q_list, R_list, b_list, d_list

        Returns
        -------
        log_w  : (N,) unnormalised log-weights
        mu_new : (N, n) updated means
        P_new  : (N, n, n) updated covariances
        """
        A_list, C_list = ssm_lists["A_list"], ssm_lists["C_list"]
        Q_list, R_list = ssm_lists["Q_list"], ssm_lists["R_list"]
        b_list, d_list = ssm_lists["b_list"], ssm_lists["d_list"]

        N, n = mu_p.shape
        m    = len(y_t)
        K    = len(A_list)

        log_w  = np.full(N, -np.inf)
        mu_new = np.empty((N, n))
        P_new  = np.empty((N, n, n))

        for k in range(K):
            mask = s_particles == k
            if not mask.any():
                continue

            A_k = A_list[k]                                          # (n, n)
            C_k = C_list[k]                                          # (m, n)
            Q_k = Q_list[k]                                          # (n, n)
            R_k = R_list[k]                                          # (m, m)
            b_k = b_list[k] if b_list is not None else np.zeros(n)  # (n,)
            d_k = d_list[k] if d_list is not None else np.zeros(m)  # (m,)

            mu_k = mu_p[mask]   # (Nk, n)
            P_k  = P_p[mask]    # (Nk, n, n)

            # Predict: mu_pred = A z_{t-1} + b,  P_pred = A P A^T + Q
            mu_pred = mu_k @ A_k.T + b_k
            AP      = np.matmul(A_k, P_k)
            P_pred  = 0.5 * (np.matmul(AP, A_k.T) + Q_k)
            P_pred += 0.5 * P_pred.transpose(0, 2, 1)

            # Innovation: v = y - C mu_pred - d,  S = C P_pred C^T + R
            v  = y_t - mu_pred @ C_k.T - d_k
            CP = np.matmul(C_k, P_pred)
            S  = np.matmul(CP, C_k.T) + R_k
            # Symmetrize and add jitter so a single near-singular matrix
            # does not kill every regime-k particle via a batch LinAlgError.
            S  = 0.5 * (S + S.transpose(0, 2, 1)) + 1e-8 * np.eye(m)

            # Log-weight: log N(y_t ; C mu_pred + d, S) via Cholesky
            try:
                L       = np.linalg.cholesky(S)                     # (Nk, m, m)
                log_det = 2.0 * np.log(np.diagonal(L, axis1=1, axis2=2)).sum(axis=1)
                z       = np.linalg.solve(L, v[:, :, None]).squeeze(-1)
                log_w[mask] = -0.5 * (m * np.log(2.0 * np.pi) + log_det + (z ** 2).sum(axis=1))
            except np.linalg.LinAlgError:
                # Cholesky failed despite jitter — leave weights at -inf.
                mu_new[mask] = mu_pred
                P_new[mask]  = P_pred
                continue

            # Update: K = P_pred C^T S^{-1},  mu_new = mu_pred + K v
            K_gain       = np.linalg.solve(S, CP).transpose(0, 2, 1)  # (Nk, n, m)
            mu_new[mask] = mu_pred + np.matmul(K_gain, v[:, :, None]).squeeze(-1)

            # Joseph form: P_new = (I - KC) P_pred (I - KC)^T + K R K^T
            KC  = np.matmul(K_gain, C_k)
            IKC = np.eye(n) - KC
            T1  = np.matmul(np.matmul(IKC, P_pred), IKC.transpose(0, 2, 1))
            T2  = np.matmul(np.matmul(K_gain, R_k),  K_gain.transpose(0, 2, 1))
            P_out        = T1 + T2
            P_new[mask]  = 0.5 * (P_out + P_out.transpose(0, 2, 1))

        return log_w, mu_new, P_new

    def run_filter(self, verbose=False):
        """
        Run the Rao-Blackwellized particle filter.

        Returns
        -------
        state_estimates  : (T, state_dim) — regime-averaged filtered state means
        particle_history : list of T (N,) int arrays — regime particles per step
        weight_history   : list of T (N,) float arrays — normalised weights
        resample_history : list of T ints (0 or 1) — resampling indicator
        loglik           : float — log p̂(y_{0:T-1})

        Also sets
        ---------
        self.regime_prob_history : (T, K) filtered regime probabilities
        """
        _t0   = time.perf_counter()
        model = self.model
        data  = np.asarray(self.data, dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        T, m = data.shape
        K    = model.n_regimes
        n    = model.state_dim
        N    = self.N_particles
        rng  = self.rng

        ssm_lists = {
            "A_list": model.A_list,
            "C_list": model.C_list,
            "Q_list": model.Q_list,
            "R_list": model.R_list,
            "b_list": getattr(model, "b_list", None),
            "d_list": getattr(model, "d_list", None),
        }

        # Initial Kalman covariance per regime: stationary if A is stable,
        # else a diagonal scaled by process noise.
        P0 = []
        for k in range(K):
            A_k, Q_k = ssm_lists["A_list"][k], ssm_lists["Q_list"][k]
            if np.all(np.abs(np.linalg.eigvals(A_k)) < 1.0):
                P0.append(symmetrize(solve_discrete_lyapunov(A_k, Q_k)))
            else:
                P0.append(np.eye(n) * (np.trace(Q_k) + 1.0))

        # Initialise particles from the stationary regime distribution.
        pi          = model.regime_probabilities_stationary
        s_particles = rng.choice(K, size=N, p=pi).astype(int)
        # Use the model's initial state mean if provided; fall back to zeros.
        m0   = getattr(model, "mu_0", None)
        mu_p = np.tile(m0, (N, 1)) if m0 is not None else np.zeros((N, n))
        P_p  = np.array([P0[s] for s in s_particles])   # (N, n, n)

        self.particle_history.clear()
        self.weight_history.clear()
        self.resample_history.clear()

        loglik              = 0.0
        state_estimates     = np.zeros((T, n))
        regime_prob_history = np.zeros((T, K))

        for t in range(T):
            y_t = data[t]

            # Fetch per-step so covariate-dependent transitions are respected.
            P_trans = model.regime_transition_matrix

            # Propagate regimes (skip t=0; particles already drawn from π).
            if t > 0:
                cum_P       = np.cumsum(P_trans[s_particles], axis=1)  # (N, K)
                u           = rng.random(N)[:, None]                    # (N, 1)
                s_particles = np.argmax(u < cum_P, axis=1).astype(int)

            # Kalman predict/update for every regime; returns per-particle log-weights.
            log_w, mu_new, P_new = self._kalman_update_step(
                y_t, s_particles, mu_p, P_p, ssm_lists
            )

            # Log-likelihood increment: log (1/N) Σ w_i
            log_sum  = logsumexp(log_w)
            loglik  += log_sum - np.log(N)

            # Normalise weights (guard against floating-point drift).
            weights  = np.exp(log_w - log_sum)
            weights /= weights.sum()

            # Regime-averaged state estimate and filtered regime probabilities.
            state_estimates[t] = (weights[:, None] * mu_new).sum(axis=0)
            for k in range(K):
                regime_prob_history[t, k] = weights[s_particles == k].sum()

            # Store history before resampling — consistent with ParticleFilter.
            self.particle_history.append(s_particles.copy())
            self.weight_history.append(weights.copy())

            # Resample when ESS drops below threshold.
            ess       = 1.0 / np.sum(weights ** 2)
            resampled = 0
            if self.resample_method is not None and ess < self.resample_threshold * N:
                idx         = self.resample_method.resample(np.arange(N), weights)
                s_particles = s_particles[idx]
                mu_new      = mu_new[idx]
                P_new       = P_new[idx]
                resampled   = 1

            self.resample_history.append(resampled)
            mu_p = mu_new
            P_p  = P_new

        self.regime_prob_history = regime_prob_history

        if verbose:
            n_resamples = sum(self.resample_history)
            elapsed     = time.perf_counter() - _t0
            print(
                f"RaoBlackwellizedParticleFilter.run_filter  {elapsed:.3f}s  "
                f"T={T}  N={N}  K={K}  resamples={n_resamples}  loglik={loglik:.2f}"
            )

        return (
            state_estimates,
            self.particle_history,
            self.weight_history,
            self.resample_history,
            loglik,
        )

    # ── smoother ──────────────────────────────────────────────────────────────

    def run_smoother(self):
        raise NotImplementedError(
            "RBPF smoother (backward simulation) is not yet implemented. "
            "Call run_filter() first to obtain the filtered distribution."
        )


# ── module-level helper ───────────────────────────────────────────────────────

def symmetrize(M: np.ndarray) -> np.ndarray:
    """(M + M^T) / 2 for a 2-D matrix."""
    return 0.5 * (M + M.T)
