import numpy as np
from scipy.linalg import cho_factor, cho_solve

from models.linear_gaussian import LinearGaussianSSM
from utils import symmetrize


class KalmanFilter:
    """
    Kalman filter and RTS smoother for LinearGaussianSSM.

    Model:
        x_t = A x_{t-1} + b + eps_t,   eps_t ~ N(0, Q)
        y_t = C x_t     + d + nu_t,    nu_t  ~ N(0, R)
        x_0 ~ N(mu_0, P_0)

    Conventions
    -----------
    - predicted_means[t] / predicted_covs[t]  : E[x_t | y_{0:t-1}]
    - filtered_means[t]  / filtered_covs[t]   : E[x_t | y_{0:t}]
    - smoothed_means[t]  / smoothed_covs[t]   : E[x_t | y_{0:T-1}]  (after run_smoother)
    """

    def __init__(self, model: LinearGaussianSSM, data):
        if not isinstance(model, LinearGaussianSSM):
            raise ValueError("KalmanFilter requires a LinearGaussianSSM instance.")

        self.model = model

        data = np.asarray(data, dtype=float)
        self.data = data[:, None] if data.ndim == 1 else data
        self.T = len(self.data)

        self.predicted_means = None
        self.predicted_covs = None
        self.filtered_means = None
        self.filtered_covs = None
        self.innovations = None
        self.innovation_covs = None
        self.loglik = None

        self.smoothed_means = None
        self.smoothed_covs = None

    def run_filter(self):
        """
        Run the Kalman filter forward.

        Returns
        -------
        filtered_means : (T, state_dim)
        filtered_covs  : (T, state_dim, state_dim)
        loglik         : float  — log p(y_{0:T-1})
        """
        mdl = self.model
        A, C, Q, R, b, d = mdl.A, mdl.C, mdl.Q, mdl.R, mdl.b, mdl.d
        n, m, T = mdl.state_dim, mdl.obs_dim, self.T

        predicted_means  = np.zeros((T, n))
        predicted_covs   = np.zeros((T, n, n))
        filtered_means   = np.zeros((T, n))
        filtered_covs    = np.zeros((T, n, n))
        innovations      = np.zeros((T, m))
        innovation_covs  = np.zeros((T, m, m))

        mu = mdl.mu_0.copy()   # prior mean  E[x_0]
        P  = mdl.P_0.copy()    # prior cov   Var[x_0]
        loglik = 0.0
        eye_n = np.eye(n)

        for t in range(T):
            # ── store predicted (t | t-1) ──────────────────────────────────────
            predicted_means[t] = mu
            predicted_covs[t]  = P

            # ── update (condition on y_t) ──────────────────────────────────────
            y_t = self.data[t]
            v   = y_t - C @ mu - d                  # innovation
            s   = symmetrize(C @ P @ C.T + R)        # innovation covariance

            innovations[t]     = v
            innovation_covs[t] = s

            cf          = cho_factor(s)
            k_gain      = cho_solve(cf, C @ P).T     # Kalman gain  (n × m)
            log_det_s   = 2.0 * np.sum(np.log(np.diag(cf[0])))
            loglik     -= 0.5 * (m * np.log(2.0 * np.pi) + log_det_s + v @ cho_solve(cf, v))

            # Joseph form: numerically stable, guarantees symmetry / PSD
            i_minus_kc = eye_n - k_gain @ C
            mu  = mu + k_gain @ v
            P   = symmetrize(i_minus_kc @ P @ i_minus_kc.T + k_gain @ R @ k_gain.T)

            filtered_means[t] = mu
            filtered_covs[t]  = P

            # ── predict for t+1 ────────────────────────────────────────────────
            if t < T - 1:
                mu = A @ mu + b
                P  = symmetrize(A @ P @ A.T + Q)

        self.predicted_means = predicted_means
        self.predicted_covs  = predicted_covs
        self.filtered_means  = filtered_means
        self.filtered_covs   = filtered_covs
        self.innovations     = innovations
        self.innovation_covs = innovation_covs
        self.loglik          = loglik

        return filtered_means, filtered_covs, loglik

    def run_smoother(self):
        """
        RTS (Rauch-Tung-Striebel) smoother.  run_filter() must be called first.

        Returns
        -------
        smoothed_means : (T, state_dim)
        smoothed_covs  : (T, state_dim, state_dim)
        """
        if self.filtered_means is None:
            raise RuntimeError("run_filter() must be called before run_smoother().")

        A = self.model.A
        T = self.T

        smoothed_means = self.filtered_means.copy()
        smoothed_covs  = self.filtered_covs.copy()

        for t in range(T - 2, -1, -1):
            p_pred = self.predicted_covs[t + 1]          # P_{t+1 | t}
            cf = cho_factor(p_pred)
            g  = cho_solve(cf, A @ self.filtered_covs[t]).T   # smoother gain (n × n)

            smoothed_means[t] += g @ (smoothed_means[t + 1] - self.predicted_means[t + 1])
            smoothed_covs[t]   = symmetrize(
                self.filtered_covs[t] + g @ (smoothed_covs[t + 1] - p_pred) @ g.T
            )

        self.smoothed_means = smoothed_means
        self.smoothed_covs  = smoothed_covs

        return smoothed_means, smoothed_covs
