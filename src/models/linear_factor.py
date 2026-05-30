import numpy as np

from models.linear_gaussian import LinearGaussianSSM


# Dynamic single-factor model with multivariate observations.
#
# One 1D latent business-cycle factor drives K observed series:
#
#   Transition:   x_t = phi * x_{t-1} + eps_t,          eps_t  ~ N(0, sigma2)
#   Observation:  y_t^(k) = mu^(k) + alpha^(k) * x_t + nu_t^(k),
#                                                        nu_t^(k) ~ N(0, tau2^(k))
#                 for k = 1, ..., K
#
# Identification: alpha^(1) = 1 (first series is the reference; its scale defines
# the factor).
#
# In LinearGaussianSSM notation (n=1, m=K):
#   A = [[phi]],  b = [0]
#   C = [[1], [alpha_2], ..., [alpha_K]]  (K×1)
#   Q = [[sigma2]]
#   R = diag(tau2_1, ..., tau2_K)         (K×K diagonal; observations are
#                                          conditionally independent given x_t)
#   d = [mu_1, ..., mu_K]                 (K-vector of intercepts)
#
# Free parameters (total 3K+1):
#   phi, sigma2, alpha_2, ..., alpha_K, tau2_1, ..., tau2_K, mu_1, ..., mu_K
#
# If the K series are pre-standardised (zero mean, unit variance), the mu^(k)
# intercepts are expected to be zero and can be fixed via MLEEstimator:
#   mle.fit(fixed_params={f'mu_{k+1}': 0.0 for k in range(K)})
#
# Flat parameter vector layout (matches params_dict ordering):
#   index 0       : phi
#   index 1       : sigma2
#   index 2..K    : alpha_2, ..., alpha_K          (K-1 free loadings)
#   index K+1..2K : tau2_1, ..., tau2_K            (K obs variances)
#   index 2K+1..3K: mu_1, ..., mu_K               (K intercepts)
class MultivariateObservationLGSSM(LinearGaussianSSM):
    """
    Multivariate-observation LG-SSM with a single 1D latent factor.

    Parameters
    ----------
    phi       : float          AR(1) coefficient, |phi| < 1
    sigma2    : float          Latent-state variance (> 0)
    alphas    : (K,) array     Factor loadings; alphas[0] must equal 1.0
    tau2s     : (K,) array     Observation variances (all > 0)
    mus       : (K,) array | None
                               Observation intercepts. None → zeros.
                               Pass zeros explicitly if you want them tracked
                               in params_dict but constrained to zero at MLE time.
    obs_names : list[str] | None
                               Names for the K observation series (display only).
    seed      : int | None
    """

    def __init__(self, phi, sigma2, alphas, tau2s, mus=None, obs_names=None, seed=None):
        alphas = np.asarray(alphas, dtype=float)
        tau2s  = np.asarray(tau2s,  dtype=float)
        K = len(alphas)
        mus = np.zeros(K) if mus is None else np.asarray(mus, dtype=float)

        if len(tau2s) != K or len(mus) != K:
            raise ValueError(f"alphas, tau2s and mus must all have length K={K}.")

        # Validate before super() for readable error messages.
        if abs(phi) >= 1:
            raise ValueError(f"phi={phi}: require |phi| < 1 for stationarity.")
        if sigma2 <= 0:
            raise ValueError(f"sigma2={sigma2}: must be positive.")
        if not np.isclose(alphas[0], 1.0):
            raise ValueError(
                f"alphas[0]={alphas[0]}: the first loading must equal 1.0 "
                "(identification constraint)."
            )
        if np.any(tau2s <= 0):
            raise ValueError(f"All tau2s must be positive; got {tau2s}.")

        self._K         = K
        self._obs_names = list(obs_names) if obs_names is not None else None

        super().__init__(
            a=np.array([[phi]]),
            c=alphas[:, None],       # (K, 1)
            q=np.array([[sigma2]]),
            r=np.diag(tau2s),
            d=mus.copy(),
            seed=seed,
        )
        # Override params_dict with scalar names (parent uses matrix keys).
        self.params_dict = self._make_params_dict(phi, sigma2, alphas, tau2s, mus)

    # ── private helpers ───────────────────────────────────────────────────────

    def _make_params_dict(self, phi, sigma2, alphas, tau2s, mus):
        K = self._K
        d = {'phi': float(phi), 'sigma2': float(sigma2)}
        for k in range(1, K):
            d[f'alpha_{k + 1}'] = float(alphas[k])
        for k in range(K):
            d[f'tau2_{k + 1}'] = float(tau2s[k])
        for k in range(K):
            d[f'mu_{k + 1}'] = float(mus[k])
        return d

    def _unpack(self, flat):
        """Unpack a flat constrained parameter list into (phi, sigma2, alphas, tau2s, mus)."""
        K = self._K
        v      = list(flat)
        phi    = float(v[0])
        sigma2 = float(v[1])
        alphas = np.array([1.0] + [float(a) for a in v[2:K + 1]])
        tau2s  = np.array([float(t) for t in v[K + 1:2 * K + 1]])
        mus    = np.array([float(m) for m in v[2 * K + 1:]])
        return phi, sigma2, alphas, tau2s, mus

    # ── scalar properties ─────────────────────────────────────────────────────

    @property
    def phi(self):    return float(self.A[0, 0])

    @property
    def sigma2(self): return float(self.Q[0, 0])

    @property
    def alphas(self): return self.C[:, 0].copy()

    @property
    def tau2s(self):  return np.diag(self.R).copy()

    @property
    def mus(self):    return self.d.copy()

    @property
    def stationary_var(self):
        return self.sigma2 / (1 - self.phi ** 2) if abs(self.phi) < 1 else np.inf

    # ── validity ──────────────────────────────────────────────────────────────

    def check_params_validity(self):
        super().check_params_validity()   # shapes, Q PSD, R PD
        if abs(self.phi) >= 1:
            raise ValueError(
                f"phi={self.phi}: require |phi| < 1 for stationarity."
            )
        if not np.isclose(self.alphas[0], 1.0):
            raise ValueError(
                f"alphas[0]={self.alphas[0]}: the first loading must equal 1.0."
            )

    # ── string representation ─────────────────────────────────────────────────

    def __repr__(self):
        return (
            f"MultivariateObservationLGSSM("
            f"K={self._K}, phi={self.phi!r}, sigma2={self.sigma2!r})"
        )

    def describe(self):
        names = self._obs_names or [f"y^({k + 1})" for k in range(self._K)]
        lines = [
            f"{self.__class__.__name__}",
            f"  Multivariate-observation LG-SSM  (K={self._K} series, 1D latent factor)",
            f"  Transition:  x_t = {self.phi:.4g} x_(t-1) + eps_t,  "
            f"eps_t ~ N(0, {self.sigma2:.4g})",
            f"  Initial:     x_0 ~ N(0, {self.stationary_var:.4g})  [stationary]",
            f"  Observations (alpha^(1) = 1.0 fixed):",
        ]
        for k in range(self._K):
            a = "1.0 [fixed]" if k == 0 else f"{self.alphas[k]:.4g}"
            mu_part = f"{self.mus[k]:.4g} + " if self.mus[k] != 0 else ""
            lines.append(
                f"    {names[k]}: {mu_part}{a} * x_t + nu^({k+1}),  "
                f"nu^({k+1}) ~ N(0, {self.tau2s[k]:.4g})"
            )
        return "\n".join(lines)

    # ── parameter interface ───────────────────────────────────────────────────

    def update_params(self, constrained_params):
        phi, sigma2, alphas, tau2s, mus = self._unpack(constrained_params)
        # Update underlying matrices via the parent (calls check_params_validity).
        super().update_params({
            'A': np.array([[phi]]),
            'C': alphas[:, None],
            'Q': np.array([[sigma2]]),
            'R': np.diag(tau2s),
            'b': np.zeros(1),
            'd': mus,
        })
        # Refresh the stationary initial distribution (phi and sigma2 may have changed).
        # LinearGaussianSSM.update_params does not do this automatically.
        self.mu_0, self.P_0 = self._stationary_distribution()
        # Restore scalar params_dict (parent overwrites with matrix keys).
        self.params_dict = self._make_params_dict(phi, sigma2, alphas, tau2s, mus)

    def unconstrain_params(self, constrained_params):
        """
        Map constrained params → unconstrained R^(3K+1).

        Transforms applied:
          phi    : arctanh     (maps (-1,1) → R)
          sigma2 : log         (maps (0,∞) → R)
          alphas : identity    (unbounded)
          tau2s  : log         (maps (0,∞) → R)
          mus    : identity    (unbounded)
        """
        phi, sigma2, alphas, tau2s, mus = self._unpack(constrained_params)
        return np.concatenate([
            [np.arctanh(phi), np.log(sigma2)],
            alphas[1:],                   # free loadings (identity)
            np.log(tau2s),                # log-transform
            mus,                          # identity
        ])

    def constrain_params(self, unconstrained_params):
        """
        Map unconstrained R^(3K+1) → constrained params list.

        Returns a flat list matching params_dict ordering:
          [phi, sigma2, alpha_2, ..., alpha_K, tau2_1, ..., tau2_K, mu_1, ..., mu_K]
        """
        K = self._K
        u = np.asarray(unconstrained_params, dtype=float)
        phi    = float(np.tanh(u[0]))
        sigma2 = float(np.exp(u[1]))
        free_a = list(u[2:K + 1])               # alpha_2..alpha_K (identity)
        tau2s  = [float(np.exp(v)) for v in u[K + 1:2 * K + 1]]
        mus    = list(u[2 * K + 1:])
        return [phi, sigma2] + free_a + tau2s + mus

    def jacobian_constrain_params(self, unconstrained_params):
        """
        Diagonal Jacobian of constrain_params at unconstrained_params.

        Shape (3K+1, 3K+1). Diagonal entries:
          phi    : 1 - tanh²(u_phi)   = 1 - phi²
          sigma2 : exp(u_sigma2)       = sigma2
          alphas : 1.0 each            (identity)
          tau2s  : exp(u_tau2_k)       = tau2_k
          mus    : 1.0 each            (identity)
        """
        K = self._K
        u = np.asarray(unconstrained_params, dtype=float)
        diag = np.concatenate([
            [1.0 - np.tanh(u[0]) ** 2,   # phi
             np.exp(u[1])],               # sigma2
            np.ones(K - 1),              # free alphas
            np.exp(u[K + 1:2 * K + 1]),  # tau2s
            np.ones(K),                  # mus
        ])
        return np.diag(diag)