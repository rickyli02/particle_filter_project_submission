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

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from estimation.particle_filter import ParticleFilter
from utils import logsumexp, timer


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

    @timer
    def run_filter(self):
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
        model = self.model
        data  = np.asarray(self.data, dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        T, m = data.shape
        K    = model.n_regimes
        n    = model.state_dim
        N    = self.N_particles
        rng  = self.rng

        P_trans = model.regime_transition_matrix          # (K, K)
        A_list  = model.A_list
        C_list  = model.C_list
        Q_list  = model.Q_list
        R_list  = model.R_list
        b_list  = getattr(model, "b_list", None)
        d_list  = getattr(model, "d_list", None)

        _b = lambda k: b_list[k] if b_list is not None else np.zeros(n)
        _d = lambda k: d_list[k] if d_list is not None else np.zeros(m)

        # ── initial Kalman covariance per regime ──────────────────────────────
        P0 = []
        for k in range(K):
            A_k = A_list[k]
            Q_k = Q_list[k]
            eigs = np.abs(np.linalg.eigvals(A_k))
            if np.all(eigs < 1.0):
                P0.append(symmetrize(solve_discrete_lyapunov(A_k, Q_k)))
            else:
                P0.append(np.eye(n) * (np.trace(Q_k) + 1.0))

        # ── initialise particles ──────────────────────────────────────────────
        pi          = model.regime_probabilities_stationary
        s_particles = rng.choice(K, size=N, p=pi).astype(int)   # (N,)
        mu_p        = np.zeros((N, n))                           # (N, n)
        P_p         = np.array([P0[s] for s in s_particles])    # (N, n, n)

        # ── clear inherited history lists ─────────────────────────────────────
        self.particle_history.clear()
        self.weight_history.clear()
        self.resample_history.clear()

        loglik              = 0.0
        state_estimates     = np.zeros((T, n))
        regime_prob_history = np.zeros((T, K))

        # ── filter loop ───────────────────────────────────────────────────────
        for t in range(T):
            y_t = data[t]   # (m,)

            # 1. Propagate regimes (skip at t=0; particles drawn from π above)
            if t > 0:
                cum_P       = np.cumsum(P_trans[s_particles], axis=1)   # (N, K)
                u           = rng.random(N)[:, None]                     # (N, 1)
                s_particles = np.argmax(u < cum_P, axis=1).astype(int)

            # 2. Per-regime vectorised Kalman predict/update + log-weights
            log_w  = np.full(N, -np.inf)
            mu_new = np.empty((N, n))
            P_new  = np.empty((N, n, n))

            for k in range(K):
                mask = s_particles == k
                if not mask.any():
                    continue

                A_k = A_list[k]   # (n, n)
                C_k = C_list[k]   # (m, n)
                Q_k = Q_list[k]   # (n, n)
                R_k = R_list[k]   # (m, m)
                b_k = _b(k)       # (n,)
                d_k = _d(k)       # (m,)

                mu_k = mu_p[mask]  # (Nk, n)
                P_k  = P_p[mask]   # (Nk, n, n)

                # ── predict ───────────────────────────────────────────────────
                # mu_pred[i] = A_k @ mu_k[i] + b_k
                mu_pred = mu_k @ A_k.T + b_k                        # (Nk, n)
                # P_pred[i] = A_k @ P_k[i] @ A_k^T + Q_k
                AP     = np.matmul(A_k, P_k)                        # (Nk, n, n)
                P_pred = np.matmul(AP, A_k.T) + Q_k                 # (Nk, n, n)
                P_pred = 0.5 * (P_pred + P_pred.transpose(0, 2, 1))

                # ── innovation ────────────────────────────────────────────────
                # v[i] = y_t - C_k @ mu_pred[i] - d_k
                v  = y_t - mu_pred @ C_k.T - d_k                    # (Nk, m)
                # S[i] = C_k @ P_pred[i] @ C_k^T + R_k
                CP = np.matmul(C_k, P_pred)                         # (Nk, m, n)
                S  = np.matmul(CP, C_k.T) + R_k                     # (Nk, m, m)
                S  = 0.5 * (S + S.transpose(0, 2, 1))

                # ── log-weights: log N(y_t ; C_k mu_pred + d_k, S) ───────────
                try:
                    L       = np.linalg.cholesky(S)                 # (Nk, m, m)
                    log_det = 2.0 * np.log(
                        np.diagonal(L, axis1=1, axis2=2)
                    ).sum(axis=1)                                    # (Nk,)
                    # solve L @ z = v  element-wise → z[i] = L[i]^{-1} v[i]
                    z = np.linalg.solve(L, v[:, :, None]).squeeze(-1)  # (Nk, m)
                    log_w[mask] = -0.5 * (
                        m * np.log(2.0 * np.pi)
                        + log_det
                        + np.sum(z ** 2, axis=1)
                    )
                except np.linalg.LinAlgError:
                    # Innovation covariance not PD — particle weight stays -inf
                    mu_new[mask] = mu_pred
                    P_new[mask]  = P_pred
                    continue

                # ── Kalman gain K = P_pred C_k^T S^{-1} ──────────────────────
                # Solve S[i] X[i] = CP[i]; then K[i] = X[i]^T
                K_gain = np.linalg.solve(S, CP).transpose(0, 2, 1)  # (Nk, n, m)

                # ── update ────────────────────────────────────────────────────
                mu_new[mask] = mu_pred + np.matmul(
                    K_gain, v[:, :, None]
                ).squeeze(-1)                                        # (Nk, n)

                # Joseph form: P = (I - KC) P_pred (I - KC)^T + K R K^T
                KC   = np.matmul(K_gain, C_k)                       # (Nk, n, n)
                IKC  = np.eye(n) - KC                               # (Nk, n, n)
                T1   = np.matmul(np.matmul(IKC, P_pred),
                                 IKC.transpose(0, 2, 1))            # (Nk, n, n)
                T2   = np.matmul(np.matmul(K_gain, R_k),
                                 K_gain.transpose(0, 2, 1))         # (Nk, n, n)
                P_out        = T1 + T2
                P_new[mask]  = 0.5 * (P_out + P_out.transpose(0, 2, 1))

            # 3. Log-likelihood contribution
            log_sum  = logsumexp(log_w)
            loglik  += log_sum - np.log(N)

            # 4. Normalise weights
            weights  = np.exp(log_w - log_sum)
            weights /= weights.sum()   # guard against floating-point drift

            # 5. Regime-averaged state and filtered regime probs
            state_estimates[t] = (weights[:, None] * mu_new).sum(axis=0)
            for k in range(K):
                regime_prob_history[t, k] = weights[s_particles == k].sum()

            # 6. Store history (before resampling — consistent with ParticleFilter)
            self.particle_history.append(s_particles.copy())
            self.weight_history.append(weights.copy())

            # 7. Resample when ESS is low
            ess       = 1.0 / np.sum(weights ** 2)
            resampled = 0
            if self.resample_method is not None and ess < self.resample_threshold * N:
                # Pass np.arange(N) to extract the resampled indices
                idx         = self.resample_method.resample(np.arange(N), weights)
                s_particles = s_particles[idx]
                mu_new      = mu_new[idx]
                P_new       = P_new[idx]
                resampled   = 1

            self.resample_history.append(resampled)
            mu_p = mu_new
            P_p  = P_new

        self.regime_prob_history = regime_prob_history

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
