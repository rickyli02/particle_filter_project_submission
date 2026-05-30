import numpy as np
from scipy.linalg import cho_factor, cho_solve

from models.linear_gaussian import LinearGaussianSSM
from utils import symmetrize


# ── Ambiguities / design decisions noted for the user ─────────────────────────
#
# 1. INITIALISATION (diffuse_var):
#    g*, ζ, u* are random walks — not stationary.  Their prior covariance is set
#    to diffuse_var * I on those components (default 1e6).  The IS-curve
#    components (x, x_{t-1}) use the same diffuse prior for simplicity; a tighter
#    AR(2)-stationary prior could be used instead.  mu_0 = 0 throughout.
#
# 2. MISSING LAGS AT t < 4:
#    The transition offset a_t needs r_{t-1} and r_{t-2}; the observation offset
#    d_t needs up to pi_{t-4}.  For t=0..3, unavailable lags fall back to pi*
#    (inflation) and the first observed value (unemployment, interest rate).
#    Alternative: drop the first 4 time steps (set burn_in=4 and start filter there).
#
# 3. TILDE_PI IN TAYLOR RULE:
#    The Taylor rule reacts to the model-implied tilde_pi_t (which contains the
#    latent u*_{t-1} component) rather than lagged observed inflation.  This
#    couples the pi_t and i_t observation equations through the same state
#    component (u*_{t-1}).  A simpler alternative: replace tilde_pi_t with
#    pi_{t-1} in the Taylor rule, making H row 4 state-independent in u*.
#
# 4. KALMAN FILTER CLASS COMPATIBILITY:
#    The generic KalmanFilter class reads model.b and model.d as time-invariant
#    offsets and will produce incorrect results for this model.  Use the model's
#    own .filter() and .smoother() methods instead.
#
# 5. TRANSITION / OBSERVATION SAMPLING (generate_data):
#    .transition() and .observation() use only the static matrices (no
#    time-varying offset) because they do not have access to the observation
#    history.  Simulating from this model correctly requires managing lags
#    externally.
#
# 6. MLE:
#    With 18 free parameters and near-singular likelihood (HLW identification
#    issues for sigma_g, sigma_zeta, c_g), full unconstrained MLE is fragile.
#    Common practice: fix or calibrate the signal-to-noise ratios sigma_u_star /
#    sigma_u and sigma_g / sigma_dY, then estimate the remaining parameters.
# ──────────────────────────────────────────────────────────────────────────────


class LinearMacroSSM(LinearGaussianSSM):
    """
    HLW-style macro state-space model.

    See linear_macro_model.md for the full structural and KF specifications.

    State (8D):
        s_t = [x_t, x_{t-1}, g_t*, zeta_t, r_t*, r*_{t-1}, u_t*, u*_{t-1}]

    Observations (4D):
        y_t = [DeltaY_t, pi_t, u_t, i_t]

    Parameters
    ----------
    phi_1, phi_2 : float
        IS-curve AR(1) and AR(2) coefficients.  Jointly constrained for
        AR(2) stability: phi_2 in (-1,1) and |phi_1 / (1 - phi_2)| < 1.
    lambda_r : float  (< 0)
        IS-curve real-rate-gap sensitivity.
    c_g : float  (> 0)
        Loading of trend growth g* onto the neutral rate r*.
    alpha_pi : float  in (0,1)
        Inflation persistence in the Phillips curve.
    beta_pi : float  (> 0)
        Unemployment-gap coefficient in the Phillips curve (positive;
        the negative sign is explicit in the equation).
    gamma : float  (> 0)
        Output-gap coefficient in Okun's law.
    rho_i : float  in [0,1)
        Monetary policy inertia in the Taylor rule.
    psi_pi : float  (> 1)
        Inflation response in the Taylor rule (Taylor principle).
    psi_x : float  (> 0)
        Output-gap response in the Taylor rule.
    sigma_x, sigma_g, sigma_zeta, sigma_u_star : float  (> 0)
        Standard deviations of the latent-state shocks.
    sigma_dY, sigma_pi, sigma_u, sigma_i : float  (> 0)
        Standard deviations of the observation noise.
    pi_star : float
        Fixed inflation target (default 2.0 pp).  Not estimated.
    diffuse_var : float
        Prior variance for all state components at t=0 (default 1e6).
    seed : int | None
    """

    # ── state / obs index constants ───────────────────────────────────────────
    _S_X     = 0   # x_t
    _S_XL    = 1   # x_{t-1}
    _S_G     = 2   # g_t*
    _S_ZETA  = 3   # zeta_t
    _S_R     = 4   # r_t*
    _S_RL    = 5   # r*_{t-1}
    _S_U     = 6   # u_t*
    _S_UL    = 7   # u*_{t-1}

    _O_DY    = 0   # DeltaY_t
    _O_PI    = 1   # pi_t
    _O_U     = 2   # u_t (observed)
    _O_I     = 3   # i_t

    def __init__(
        self,
        phi_1, phi_2, lambda_r,
        c_g,
        alpha_pi, beta_pi,
        gamma,
        rho_i, psi_pi, psi_x,
        sigma_x, sigma_g, sigma_zeta, sigma_u_star,
        sigma_dY, sigma_pi, sigma_u, sigma_i,
        pi_star=2.0,
        diffuse_var=1e6,
        seed=None,
    ):
        # Validate before super() for clear error messages.
        _pos = dict(c_g=c_g, beta_pi=beta_pi, gamma=gamma, psi_x=psi_x,
                    sigma_x=sigma_x, sigma_g=sigma_g, sigma_zeta=sigma_zeta,
                    sigma_u_star=sigma_u_star, sigma_dY=sigma_dY,
                    sigma_pi=sigma_pi, sigma_u=sigma_u, sigma_i=sigma_i)
        for name, val in _pos.items():
            if float(val) <= 0:
                raise ValueError(f"{name}={val}: must be positive.")
        if float(lambda_r) >= 0:
            raise ValueError(f"lambda_r={lambda_r}: must be negative (IS-curve rate sensitivity).")
        if not (0 < float(alpha_pi) < 1):
            raise ValueError(f"alpha_pi={alpha_pi}: must be in (0, 1).")
        if not (0 <= float(rho_i) < 1):
            raise ValueError(f"rho_i={rho_i}: must be in [0, 1).")
        if float(psi_pi) <= 1:
            raise ValueError(f"psi_pi={psi_pi}: must be > 1 (Taylor principle).")

        # Store scalar parameters.
        self._phi_1       = float(phi_1)
        self._phi_2       = float(phi_2)
        self._lambda_r    = float(lambda_r)
        self._c_g         = float(c_g)
        self._alpha_pi    = float(alpha_pi)
        self._beta_pi     = float(beta_pi)
        self._gamma       = float(gamma)
        self._rho_i       = float(rho_i)
        self._psi_pi      = float(psi_pi)
        self._psi_x       = float(psi_x)
        self._sigma_x     = float(sigma_x)
        self._sigma_g     = float(sigma_g)
        self._sigma_zeta  = float(sigma_zeta)
        self._sigma_u_star= float(sigma_u_star)
        self._sigma_dY    = float(sigma_dY)
        self._sigma_pi    = float(sigma_pi)
        self._sigma_u     = float(sigma_u)
        self._sigma_i     = float(sigma_i)
        self.pi_star      = float(pi_star)
        self._diffuse_var = float(diffuse_var)

        A, Q, H, R = self._build_matrices()
        mu_0 = np.zeros(8)
        P_0  = diffuse_var * np.eye(8)

        super().__init__(
            a=A, c=H, q=Q, r=R,
            b=np.zeros(8), d=np.zeros(4),
            mu_0=mu_0, p_0=P_0,
            seed=seed,
        )
        self.params_dict = self._make_params_dict()

    # ── private helpers ───────────────────────────────────────────────────────

    def _make_params_dict(self):
        return {
            'phi_1': self._phi_1, 'phi_2': self._phi_2,
            'lambda_r': self._lambda_r, 'c_g': self._c_g,
            'alpha_pi': self._alpha_pi, 'beta_pi': self._beta_pi,
            'gamma': self._gamma,
            'rho_i': self._rho_i, 'psi_pi': self._psi_pi, 'psi_x': self._psi_x,
            'sigma_x': self._sigma_x, 'sigma_g': self._sigma_g,
            'sigma_zeta': self._sigma_zeta, 'sigma_u_star': self._sigma_u_star,
            'sigma_dY': self._sigma_dY, 'sigma_pi': self._sigma_pi,
            'sigma_u': self._sigma_u, 'sigma_i': self._sigma_i,
        }

    def _build_matrices(self):
        """Construct the time-invariant A, Q, H, R matrices from scalar parameters."""
        phi_1, phi_2   = self._phi_1, self._phi_2
        lr              = self._lambda_r
        cg              = self._c_g
        alp, bpi        = self._alpha_pi, self._beta_pi
        gam             = self._gamma
        ri, ppi, px     = self._rho_i, self._psi_pi, self._psi_x
        sx, sg, sz, su  = self._sigma_x, self._sigma_g, self._sigma_zeta, self._sigma_u_star
        sdY, spi, su_o, si = self._sigma_dY, self._sigma_pi, self._sigma_u, self._sigma_i

        # ── Transition matrix A (8×8) ─────────────────────────────────────────
        A = np.zeros((8, 8))
        # Row 0  x_t : IS curve (state-dependent part only; observable real-rate
        #              terms go into the time-varying offset a_t).
        A[0, self._S_X]  = phi_1
        A[0, self._S_XL] = phi_2
        A[0, self._S_R]  = -lr / 2   # -lambda_r/2 > 0  (lambda_r < 0)
        A[0, self._S_RL] = -lr / 2
        # Row 1  x_{t-1}: copy of previous x_t
        A[1, self._S_X]  = 1.0
        # Row 2  g*: random walk
        A[2, self._S_G]  = 1.0
        # Row 3  zeta: random walk
        A[3, self._S_ZETA] = 1.0
        # Row 4  r_t* = c_g * g_t* + zeta_t  →  expressed via previous-step states
        A[4, self._S_G]    = cg
        A[4, self._S_ZETA] = 1.0
        # Row 5  r*_{t-1}: copy of previous r_t*
        A[5, self._S_R]  = 1.0
        # Row 6  u_t*: random walk
        A[6, self._S_U]  = 1.0
        # Row 7  u*_{t-1}: copy of previous u_t*
        A[7, self._S_U]  = 1.0

        # ── Process noise Q (8×8, rank 4) ────────────────────────────────────
        # r_t* shock = c_g * eps_g + eps_zeta → correlated (g*, zeta, r*) block.
        # Rows / cols for x_{t-1}, r*_{t-1}, u*_{t-1} are zero (no independent noise).
        sg2, sz2 = sg**2, sz**2
        Q = np.zeros((8, 8))
        Q[self._S_X,    self._S_X]    = sx**2
        Q[self._S_G,    self._S_G]    = sg2
        Q[self._S_ZETA, self._S_ZETA] = sz2
        Q[self._S_R,    self._S_R]    = cg**2 * sg2 + sz2
        Q[self._S_G,    self._S_R]    = cg * sg2      # off-diagonal g*/r* block
        Q[self._S_R,    self._S_G]    = cg * sg2
        Q[self._S_ZETA, self._S_R]    = sz2           # off-diagonal zeta/r* block
        Q[self._S_R,    self._S_ZETA] = sz2
        Q[self._S_U,    self._S_U]    = su**2

        # ── Observation matrix H (4×8) ────────────────────────────────────────
        # Lagged-observable terms (pi lags, u lag, i lag) are handled in d_t,
        # not in H.  Only the state-dependent pieces appear here.
        H = np.zeros((4, 8))
        # Row 0  DeltaY_t = x_t - x_{t-1} + g_t*
        H[self._O_DY, self._S_X]   =  1.0
        H[self._O_DY, self._S_XL]  = -1.0
        H[self._O_DY, self._S_G]   =  1.0
        # Row 1  pi_t: only state piece is +beta_pi * u*_{t-1} (from tilde_pi_t)
        H[self._O_PI, self._S_UL]  = bpi
        # Row 2  u_t (Okun): u_t* - gamma * x_t
        H[self._O_U, self._S_X]    = -gam
        H[self._O_U, self._S_U]    =  1.0
        # Row 3  i_t (Taylor): (1-rho_i)*psi_x*x_t + (1-rho_i)*r_t*
        #         + (1-rho_i)*psi_pi*beta_pi * u*_{t-1}   [from expanding tilde_pi_t]
        H[self._O_I, self._S_X]    = (1 - ri) * px
        H[self._O_I, self._S_R]    = (1 - ri)
        H[self._O_I, self._S_UL]   = (1 - ri) * ppi * bpi

        # ── Observation noise R (4×4, diagonal) ──────────────────────────────
        R = np.diag([sdY**2, spi**2, su_o**2, si**2])

        return A, Q, H, R

    def _compute_offsets(self, data, t):
        """
        Compute time-varying offsets a_t (transition) and d_t (observation).

        a_t[0] = lambda_r * (r_{t-1} + r_{t-2}) / 2,  r_{t-k} = i_{t-k} - pi_{t-k}
        d_t[1] = alpha_pi*pi_{t-1} + (1-alpha_pi)*pi_bar_{t-2:4} - beta_pi*u_{t-1}
        d_t[3] = rho_i*i_{t-1} + (1-rho_i)*[(1-psi_pi)*pi* + psi_pi*tilde_pi_obs]

        Missing lags (t < 4) fall back to pi_star for inflation and the first
        row of data for u and i  (see ambiguity note 2 at the top of this file).
        """
        lr   = self._lambda_r
        alp  = self._alpha_pi
        bpi  = self._beta_pi
        ri   = self._rho_i
        ppi  = self._psi_pi
        ps   = self.pi_star

        u0 = float(data[0, self._O_U])   # fallback for missing u lags
        i0 = float(data[0, self._O_I])   # fallback for missing i lags

        def _lag(col, k):
            idx = t - k
            if idx >= 0:
                return float(data[idx, col])
            if col == self._O_PI:
                return ps
            if col == self._O_U:
                return u0
            return i0   # col == _O_I

        pi1 = _lag(self._O_PI, 1)
        pi2 = _lag(self._O_PI, 2)
        pi3 = _lag(self._O_PI, 3)
        pi4 = _lag(self._O_PI, 4)
        u1  = _lag(self._O_U,  1)
        i1  = _lag(self._O_I,  1)
        i2  = _lag(self._O_I,  2)

        pi_bar        = (pi2 + pi3 + pi4) / 3.0
        pi_tilde_obs  = alp * pi1 + (1 - alp) * pi_bar - bpi * u1
        r1 = i1 - pi1
        r2 = i2 - pi2

        a_t = np.zeros(8)
        a_t[self._S_X] = lr * (r1 + r2) / 2.0

        d_t = np.zeros(4)
        d_t[self._O_PI] = pi_tilde_obs
        d_t[self._O_I]  = (ri * i1
                           + (1 - ri) * ((1 - ppi) * ps + ppi * pi_tilde_obs))

        return a_t, d_t

    # ── validity ──────────────────────────────────────────────────────────────

    def check_params_validity(self):
        super().check_params_validity()   # shape, Q PSD, R PD
        if self._lambda_r >= 0:
            raise ValueError(f"lambda_r={self._lambda_r}: must be negative.")
        if not (0 < self._alpha_pi < 1):
            raise ValueError(f"alpha_pi={self._alpha_pi}: must be in (0, 1).")
        if not (0 <= self._rho_i < 1):
            raise ValueError(f"rho_i={self._rho_i}: must be in [0, 1).")
        if self._psi_pi <= 1:
            raise ValueError(f"psi_pi={self._psi_pi}: must be > 1.")

    # ── log-likelihood (overrides parent to use time-varying offsets) ─────────

    def log_likelihood(self, data):
        """
        Exact log p(y_{0:T-1} | theta) via Kalman filter with time-varying offsets.

        Parameters
        ----------
        data : (T, 4) array — columns [DeltaY_t, pi_t, u_t, i_t]

        Returns
        -------
        float
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] != 4:
            raise ValueError(f"data must be shape (T, 4); got {data.shape}.")
        T   = len(data)
        A, C, Q, R = self.A, self.C, self.Q, self.R
        n, m = self.state_dim, self.obs_dim
        eye_n = np.eye(n)

        mu     = self.mu_0.copy()
        P      = self.P_0.copy()
        loglik = 0.0

        for t in range(T):
            _, d_t = self._compute_offsets(data, t)

            v  = data[t] - C @ mu - d_t
            S  = symmetrize(C @ P @ C.T + R)
            cf = cho_factor(S)
            loglik -= 0.5 * (
                m * np.log(2.0 * np.pi)
                + 2.0 * np.sum(np.log(np.diag(cf[0])))
                + v @ cho_solve(cf, v)
            )

            K   = cho_solve(cf, C @ P).T
            imc = eye_n - K @ C
            mu  = mu + K @ v
            P   = symmetrize(imc @ P @ imc.T + K @ R @ K.T)

            if t < T - 1:
                a_next, _ = self._compute_offsets(data, t + 1)
                mu = A @ mu + a_next
                P  = symmetrize(A @ P @ A.T + Q)

        return loglik

    # ── filter / smoother ─────────────────────────────────────────────────────

    def filter(self, data):
        """
        Kalman filter with time-varying offsets.

        Use this instead of the generic KalmanFilter class, which does not
        support time-varying offsets (see ambiguity note 4).

        Parameters
        ----------
        data : (T, 4) array

        Returns
        -------
        filtered_means  : (T, 8)
        filtered_covs   : (T, 8, 8)
        loglik          : float
        predicted_means : (T, 8)
        predicted_covs  : (T, 8, 8)
        innovations     : (T, 4)
        innovation_covs : (T, 4, 4)
        """
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or data.shape[1] != 4:
            raise ValueError(f"data must be shape (T, 4); got {data.shape}.")
        T   = len(data)
        A, C, Q, R = self.A, self.C, self.Q, self.R
        n, m = self.state_dim, self.obs_dim
        eye_n = np.eye(n)

        pred_m  = np.zeros((T, n))
        pred_P  = np.zeros((T, n, n))
        filt_m  = np.zeros((T, n))
        filt_P  = np.zeros((T, n, n))
        innov   = np.zeros((T, m))
        innov_S = np.zeros((T, m, m))

        mu     = self.mu_0.copy()
        P      = self.P_0.copy()
        loglik = 0.0

        for t in range(T):
            _, d_t = self._compute_offsets(data, t)

            pred_m[t] = mu
            pred_P[t] = P

            v  = data[t] - C @ mu - d_t
            S  = symmetrize(C @ P @ C.T + R)
            innov[t]   = v
            innov_S[t] = S

            cf = cho_factor(S)
            loglik -= 0.5 * (
                m * np.log(2.0 * np.pi)
                + 2.0 * np.sum(np.log(np.diag(cf[0])))
                + v @ cho_solve(cf, v)
            )

            K   = cho_solve(cf, C @ P).T
            imc = eye_n - K @ C
            mu  = mu + K @ v
            P   = symmetrize(imc @ P @ imc.T + K @ R @ K.T)

            filt_m[t] = mu
            filt_P[t] = P

            if t < T - 1:
                a_next, _ = self._compute_offsets(data, t + 1)
                mu = A @ mu + a_next
                P  = symmetrize(A @ P @ A.T + Q)

        return filt_m, filt_P, loglik, pred_m, pred_P, innov, innov_S

    def smoother(self, data):
        """
        RTS smoother.  Runs the Kalman filter internally.

        Parameters
        ----------
        data : (T, 4) array

        Returns
        -------
        smoothed_means : (T, 8)
        smoothed_covs  : (T, 8, 8)
        filtered_means : (T, 8)
        filtered_covs  : (T, 8, 8)
        loglik         : float
        """
        filt_m, filt_P, loglik, pred_m, pred_P, _, _ = self.filter(data)
        T = len(data)
        A = self.A

        sm_m = filt_m.copy()
        sm_P = filt_P.copy()

        for t in range(T - 2, -1, -1):
            # Solve P_{t+1|t} G' = A P_{t|t} via lstsq (robust to near-singular
            # predicted covs that arise from rank-deficient Q and auxiliary states).
            G = np.linalg.lstsq(
                pred_P[t + 1], A @ filt_P[t], rcond=None
            )[0].T
            sm_m[t] += G @ (sm_m[t + 1] - pred_m[t + 1])
            sm_P[t]  = symmetrize(
                filt_P[t] + G @ (sm_P[t + 1] - pred_P[t + 1]) @ G.T
            )

        return sm_m, sm_P, filt_m, filt_P, loglik

    # ── parameter interface ───────────────────────────────────────────────────

    def update_params(self, constrained_params):
        (phi_1, phi_2, lambda_r, c_g, alpha_pi, beta_pi, gamma,
         rho_i, psi_pi, psi_x,
         sigma_x, sigma_g, sigma_zeta, sigma_u_star,
         sigma_dY, sigma_pi, sigma_u, sigma_i) = constrained_params

        self._phi_1        = float(phi_1)
        self._phi_2        = float(phi_2)
        self._lambda_r     = float(lambda_r)
        self._c_g          = float(c_g)
        self._alpha_pi     = float(alpha_pi)
        self._beta_pi      = float(beta_pi)
        self._gamma        = float(gamma)
        self._rho_i        = float(rho_i)
        self._psi_pi       = float(psi_pi)
        self._psi_x        = float(psi_x)
        self._sigma_x      = float(sigma_x)
        self._sigma_g      = float(sigma_g)
        self._sigma_zeta   = float(sigma_zeta)
        self._sigma_u_star = float(sigma_u_star)
        self._sigma_dY     = float(sigma_dY)
        self._sigma_pi     = float(sigma_pi)
        self._sigma_u      = float(sigma_u)
        self._sigma_i      = float(sigma_i)

        A, Q, H, R = self._build_matrices()
        super().update_params({
            'A': A, 'C': H, 'Q': Q, 'R': R,
            'b': np.zeros(8), 'd': np.zeros(4),
        })
        self.params_dict = self._make_params_dict()

    def unconstrain_params(self, constrained_params):
        """
        Map constrained → unconstrained R^18.

        Transforms:
          phi_1, phi_2  : PACF reparameterisation (guarantees AR(2) stability)
                          kappa_2 = phi_2,  kappa_1 = phi_1 / (1 - phi_2)
                          u_k = arctanh(kappa_k)
          lambda_r      : log(-lambda_r)       (lambda_r < 0)
          c_g, beta_pi, gamma, psi_x : log     (positive)
          alpha_pi, rho_i : logit              (in (0,1) / [0,1))
          psi_pi        : log(psi_pi - 1)      (psi_pi > 1)
          sigma_*       : log                  (positive)
        """
        (phi_1, phi_2, lambda_r, c_g, alpha_pi, beta_pi, gamma,
         rho_i, psi_pi, psi_x,
         sigma_x, sigma_g, sigma_zeta, sigma_u_star,
         sigma_dY, sigma_pi, sigma_u, sigma_i) = constrained_params

        kappa_2 = float(phi_2)
        kappa_1 = float(phi_1) / (1.0 - kappa_2)
        u_phi_1 = np.arctanh(np.clip(kappa_1, -1 + 1e-7, 1 - 1e-7))
        u_phi_2 = np.arctanh(np.clip(kappa_2, -1 + 1e-7, 1 - 1e-7))

        return np.array([
            u_phi_1, u_phi_2,
            np.log(-float(lambda_r)),
            np.log(float(c_g)),
            np.log(float(alpha_pi) / (1.0 - float(alpha_pi))),
            np.log(float(beta_pi)),
            np.log(float(gamma)),
            np.log(float(rho_i) / (1.0 - float(rho_i))) if rho_i > 0 else -30.0,
            np.log(float(psi_pi) - 1.0),
            np.log(float(psi_x)),
            np.log(float(sigma_x)),   np.log(float(sigma_g)),
            np.log(float(sigma_zeta)), np.log(float(sigma_u_star)),
            np.log(float(sigma_dY)),  np.log(float(sigma_pi)),
            np.log(float(sigma_u)),   np.log(float(sigma_i)),
        ])

    def constrain_params(self, unconstrained_params):
        """
        Map unconstrained R^18 → constrained params list.

        Returns flat list matching params_dict ordering.
        """
        u = np.asarray(unconstrained_params, dtype=float)

        kappa_1 = float(np.tanh(u[0]))
        kappa_2 = float(np.tanh(u[1]))
        phi_2   = kappa_2
        phi_1   = kappa_1 * (1.0 - phi_2)

        def _sigmoid(x):
            return float(1.0 / (1.0 + np.exp(-x)))

        return [
            phi_1, phi_2,
            -float(np.exp(u[2])),           # lambda_r  < 0
            float(np.exp(u[3])),            # c_g
            _sigmoid(u[4]),                 # alpha_pi  in (0,1)
            float(np.exp(u[5])),            # beta_pi
            float(np.exp(u[6])),            # gamma
            _sigmoid(u[7]),                 # rho_i     in [0,1)
            1.0 + float(np.exp(u[8])),      # psi_pi    > 1
            float(np.exp(u[9])),            # psi_x
            float(np.exp(u[10])),           # sigma_x
            float(np.exp(u[11])),           # sigma_g
            float(np.exp(u[12])),           # sigma_zeta
            float(np.exp(u[13])),           # sigma_u_star
            float(np.exp(u[14])),           # sigma_dY
            float(np.exp(u[15])),           # sigma_pi
            float(np.exp(u[16])),           # sigma_u
            float(np.exp(u[17])),           # sigma_i
        ]

    def jacobian_constrain_params(self, unconstrained_params):
        """
        Jacobian of constrain_params.  Not diagonal (phi_1 and phi_2 are coupled
        through the PACF reparameterisation), so the full (18×18) matrix is needed.
        Currently raises NotImplementedError; use numerical differentiation via
        MLEEstimator.compute_std_errors() which does not rely on this method.
        """
        raise NotImplementedError(
            "jacobian_constrain_params: phi_1/phi_2 coupling makes the Jacobian "
            "non-diagonal.  MLEEstimator.compute_std_errors() uses numerical "
            "finite-differences and does not call this method."
        )

    # ── string representation ─────────────────────────────────────────────────

    def __repr__(self):
        return (
            f"LinearMacroSSM("
            f"phi_1={self._phi_1:.4g}, phi_2={self._phi_2:.4g}, "
            f"lambda_r={self._lambda_r:.4g}, c_g={self._c_g:.4g})"
        )

    def describe(self):
        lines = [
            "LinearMacroSSM  (HLW macro state-space model)",
            "  State (8D): [x_t, x_{t-1}, g_t*, zeta_t, r_t*, r*_{t-1}, u_t*, u*_{t-1}]",
            "  Obs   (4D): [DeltaY_t, pi_t, u_t, i_t]",
            "",
            f"  IS curve:   phi_1={self._phi_1:.4g}, phi_2={self._phi_2:.4g}, "
            f"lambda_r={self._lambda_r:.4g}",
            f"  Neutral r*: c_g={self._c_g:.4g}",
            f"  Phillips:   alpha_pi={self._alpha_pi:.4g}, beta_pi={self._beta_pi:.4g}",
            f"  Okun:       gamma={self._gamma:.4g}",
            f"  Taylor:     rho_i={self._rho_i:.4g}, psi_pi={self._psi_pi:.4g}, "
            f"psi_x={self._psi_x:.4g},  pi*={self.pi_star:.4g} (fixed)",
            "",
            f"  Latent noise SDs:  sigma_x={self._sigma_x:.4g}, "
            f"sigma_g={self._sigma_g:.4g}, sigma_zeta={self._sigma_zeta:.4g}, "
            f"sigma_u*={self._sigma_u_star:.4g}",
            f"  Obs noise SDs:     sigma_dY={self._sigma_dY:.4g}, "
            f"sigma_pi={self._sigma_pi:.4g}, sigma_u={self._sigma_u:.4g}, "
            f"sigma_i={self._sigma_i:.4g}",
        ]
        return "\n".join(lines)

    # ── sampling (time-invariant fallbacks, no offset applied) ───────────────
    # These use the static matrices without time-varying offsets.
    # For correct simulation, manage lags externally (see ambiguity note 5).

    def sample_initial_distribution(self):
        return self.rng.multivariate_normal(self.mu_0, self.P_0)

    def transition(self, x_prev):
        x_prev = np.asarray(x_prev, dtype=float)
        return self.A @ x_prev + self.rng.multivariate_normal(
            np.zeros(self.state_dim), self.Q
        )

    def observation(self, x):
        x = np.asarray(x, dtype=float)
        return self.C @ x + self.rng.multivariate_normal(
            np.zeros(self.obs_dim), self.R
        )

    def log_observation_density(self, y, x):
        from scipy.stats import multivariate_normal
        mean = self.C @ np.asarray(x, dtype=float)
        return multivariate_normal.logpdf(y, mean=mean, cov=self.R)
