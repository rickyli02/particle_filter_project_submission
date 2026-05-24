"""
Kim (1994) Approximate Filter for Markov-Switching Linear Gaussian SSMs
========================================================================
Analytically marginalizes the continuous latent state via per-regime Kalman
filters while tracking discrete regime probabilities exactly up to the
collapsing approximation.

Algorithm sketch (one time step)
---------------------------------
  Maintain K filtered Gaussians:  (prob[k], mu[k], P_cov[k])  for k = 0...K-1

  Predict:  for each pair (s_{t-1}=j, s_t=i):
              mu_pred[j,i]  = A_i @ mu[j]
              P_pred[j,i]   = A_i @ P_cov[j] @ A_i' + Q_i

  Update:   for each pair (j, i):
              innovation + Kalman update → mu_upd[j,i], P_upd[j,i]
              log f(y_t | j, i) via Cholesky of innovation covariance S[j,i]

  Joint:    log_joint[j,i] = log f + log P_trans[j,i] + log prob[j]

  Collapse: marginalise j → filtered prob[i]; then moment-match to reduce
            K^2 Gaussians back to K (Kim's approximation).

Compatible model must expose
-----------------------------
  n_regimes, regime_transition_matrix, regime_probabilities_stationary
  A_list, C_list, Q_list, R_list          (per-regime matrices)
  state_dim, obs_dim
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_discrete_lyapunov

from models.base import StateSpaceModel
from utils import symmetrize, _logsumexp2d


# ── Kim filter ────────────────────────────────────────────────────────────────

class KimFilter:
    """
    Kim (1994) approximate filter for Markov-switching linear Gaussian SSMs.

    Parameters
    ----------
    model : StateSpaceModel
        Must expose n_regimes, regime_transition_matrix,
        regime_probabilities_stationary, A_list, C_list, Q_list, R_list.
    data  : array-like, shape (T, obs_dim) or (T,) for univariate
    """

    _REQUIRED = (
        "n_regimes",
        "regime_transition_matrix",
        "regime_probabilities_stationary",
        "A_list", "C_list", "Q_list", "R_list",
    )

    def __init__(self, model: StateSpaceModel, data):
        for attr in self._REQUIRED:
            if not hasattr(model, attr):
                raise ValueError(
                    f"KimFilter requires model.{attr}; "
                    f"{type(model).__name__} does not provide it."
                )
        self.model = model
        self.data  = np.asarray(data, dtype=float)
        if self.data.ndim == 1:
            self.data = self.data[:, None]

        # Populated by run_filter; used by run_smoother
        self.prob_hist: np.ndarray | None = None   # (T, K)
        self.mu_hist:   np.ndarray | None = None   # (T, K, n)
        self.P_hist:    np.ndarray | None = None   # (T, K, n, n)
        self.loglik:    float | None      = None

    def __repr__(self) -> str:
        T = len(self.data)
        return (
            f"KimFilter(model={self.model!r}, "
            f"T={T}, K={self.model.n_regimes})"
        )

    # ── filter ────────────────────────────────────────────────────────────────

    def run_filter(self):
        """
        Run the Kim approximate forward filter.

        Returns
        -------
        filtered_means : (T, state_dim)
            E[x_t | y_{0:t}] averaged over regimes.
        filtered_probs : (T, K)
            Pr(s_t = i | y_{0:t}) for each regime i.
        loglik : float
            Approximate log p(y_{0:T-1}).
        """
        model   = self.model
        K       = model.n_regimes
        n       = model.state_dim
        m       = self.data.shape[1]
        T       = len(self.data)
        P_trans = model.regime_transition_matrix   # (K, K), rows sum to 1
        eye_n   = np.eye(n)

        # ── initialise ────────────────────────────────────────────────────────
        prob  = model.regime_probabilities_stationary.copy().astype(float)
        mu    = np.zeros((K, n))
        P_cov = np.zeros((K, n, n))

        for k in range(K):
            A_k = model.A_list[k]
            Q_k = model.Q_list[k]
            eigs = np.abs(np.linalg.eigvals(A_k))
            if np.all(eigs < 1.0):
                P_cov[k] = solve_discrete_lyapunov(A_k, Q_k)
            else:
                # unstable dynamics: use diffuse prior
                P_cov[k] = np.eye(n) * (np.trace(Q_k) + 1.0)

        # ── storage ───────────────────────────────────────────────────────────
        prob_hist = np.zeros((T, K))
        mu_hist   = np.zeros((T, K, n))
        P_hist    = np.zeros((T, K, n, n))
        loglik    = 0.0

        # ── filter loop ───────────────────────────────────────────────────────
        for t in range(T):
            y_t = self.data[t]   # (m,)

            log_joint = np.full((K, K), -np.inf)   # log_joint[j, i]
            mu_upd    = np.zeros((K, K, n))         # mu_upd[j, i]
            P_upd     = np.zeros((K, K, n, n))      # P_upd[j, i]

            for i in range(K):
                A_i = model.A_list[i]
                C_i = model.C_list[i]
                Q_i = model.Q_list[i]
                R_i = model.R_list[i]

                for j in range(K):
                    # ── predict ───────────────────────────────────────────────
                    mu_pred = A_i @ mu[j]
                    P_pred  = symmetrize(A_i @ P_cov[j] @ A_i.T + Q_i)

                    # ── innovation and log-likelihood factor ───────────────────
                    v   = y_t - C_i @ mu_pred
                    S   = symmetrize(C_i @ P_pred @ C_i.T + R_i)
                    cf  = cho_factor(S)
                    log_det_s = 2.0 * np.sum(np.log(np.diag(cf[0])))
                    log_f = -0.5 * (
                        m * np.log(2.0 * np.pi)
                        + log_det_s
                        + v @ cho_solve(cf, v)
                    )

                    # ── Kalman update (Joseph form) ────────────────────────────
                    K_gain        = cho_solve(cf, C_i @ P_pred).T    # (n, m)
                    I_KC          = eye_n - K_gain @ C_i
                    mu_upd[j, i]  = mu_pred + K_gain @ v
                    P_upd[j, i]   = symmetrize(
                        I_KC @ P_pred @ I_KC.T + K_gain @ R_i @ K_gain.T
                    )

                    # log Pr(s_{t-1}=j, s_t=i, y_t | y_{0:t-1})
                    log_p_trans = np.log(max(P_trans[j, i], 1e-300))
                    log_p_prev  = np.log(max(prob[j],       1e-300))
                    log_joint[j, i] = log_f + log_p_trans + log_p_prev

            # ── log-likelihood contribution ───────────────────────────────────
            log_sum = _logsumexp2d(log_joint)
            loglik += log_sum

            # ── normalise → Pr(s_{t-1}=j, s_t=i | y_{0:t}) ──────────────────
            joint = np.exp(log_joint - log_sum)    # (K, K)

            # ── filtered regime probabilities: Pr(s_t=i | y_{0:t}) ────────────
            prob_new = joint.sum(axis=0)            # (K,)  marginalise over j
            prob_new = np.maximum(prob_new, 1e-300)
            prob_new /= prob_new.sum()

            # ── collapse: reduce K^2 Gaussians → K (Kim approximation) ────────
            mu_new  = np.zeros((K, n))
            P_new   = np.zeros((K, n, n))
            for i in range(K):
                w = joint[:, i] / prob_new[i]      # (K,) weights over j
                mu_new[i] = w @ mu_upd[:, i]       # sum_j w[j] * mu_upd[j,i]
                for j in range(K):
                    diff        = (mu_upd[j, i] - mu_new[i])[:, None]
                    P_new[i]   += w[j] * (P_upd[j, i] + diff @ diff.T)
                P_new[i] = symmetrize(P_new[i])

            # ── store and advance ─────────────────────────────────────────────
            prob_hist[t] = prob_new
            mu_hist[t]   = mu_new
            P_hist[t]    = P_new

            prob  = prob_new
            mu    = mu_new
            P_cov = P_new

        # ── regime-averaged filtered state means ──────────────────────────────
        # E[x_t | y_{0:t}] = Σ_i Pr(s_t=i | y_{0:t}) * mu[t,i]
        filtered_means = np.einsum("ti,tin->tn", prob_hist, mu_hist)

        self.prob_hist = prob_hist
        self.mu_hist   = mu_hist
        self.P_hist    = P_hist
        self.loglik    = loglik

        return filtered_means, prob_hist, loglik

    # ── smoother ──────────────────────────────────────────────────────────────

    def run_smoother(self):
        # Kim smoother (backward pass) — not yet implemented
        raise NotImplementedError(
            "run_smoother is not yet implemented. Call run_filter() first."
        )
