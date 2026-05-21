import numpy as np
from regime_switching import systematic_resample, stationary_regime_probs, build_matrices, default_initial_state, log_normal_pdf_scalar, logsumexp


def kalman_predict_update(m_prev, C_prev, y_t, regime_j, theta):
    """
    One Kalman predict/update step conditional on current regime j.

    Parameters
    ----------
    m_prev : array, shape (2,)
        Previous mean of a_{t-1} = [x_{t-1}, x_{t-2}]'.

    C_prev : array, shape (2, 2)
        Previous covariance.

    y_t : float
        Current observation, real GDP growth.

    regime_j : int
        Current regime index, 0 or 1.

    theta : array-like
        [p11, p22, phi, sigma1, sigma2, g1, g2, mu, tau]

    Returns
    -------
    m_new : array, shape (2,)
    C_new : array, shape (2, 2)
    log_pred_density : float
        log p(y_t | past, current regime path)
    y_mean : float
    y_var : float
    """
    p11, p22, phi, sigma1, sigma2, g1, g2, mu, tau = theta

    sigmas = np.array([sigma1, sigma2])
    gstars = np.array([g1, g2])

    sigma_j = sigmas[regime_j]
    g_j = gstars[regime_j]

    F, Q, H = build_matrices(phi, sigma_j, mu)

    # Prediction
    m_pred = F @ m_prev
    C_pred = F @ C_prev @ F.T + Q

    # Observation prediction
    y_mean = g_j + H @ m_pred
    y_var = float(H @ C_pred @ H.T + tau ** 2)

    log_pred_density = log_normal_pdf_scalar(y_t, y_mean, y_var)

    # Kalman update
    if not np.isfinite(log_pred_density):
        return m_pred, C_pred, -np.inf, y_mean, y_var

    K = (C_pred @ H.T) / y_var
    innovation = y_t - y_mean

    m_new = m_pred + K * innovation
    C_new = C_pred - np.outer(K, K) * y_var

    # Symmetrize for numerical stability
    C_new = 0.5 * (C_new + C_new.T)

    return m_new, C_new, log_pred_density, y_mean, y_var


def kim_filter_regime_growth(
    y,
    theta,
    m_init=None,
    C_init=None,
):
    """
    Kim filter approximation for the model:

        x_t = phi x_{t-1} + sigma_{s_t} eps_t

        y_t = g^*_{s_t} + mu (x_t - x_{t-1}) + tau eta_t

    Augmented state:
        a_t = [x_t, x_{t-1}]'

    The filter keeps one Gaussian approximation per current regime.

    Parameters
    ----------
    y : array-like, shape (T,)
        Observed real GDP growth.

    theta : array-like, shape (9,)
        [p11, p22, phi, sigma1, sigma2, g1, g2, mu, tau]

    Returns
    -------
    out : dict
        loglik : approximate Kim-filter log likelihood
        regime_probs : filtered regime probabilities, shape (T, 2)
        state_mean_by_regime : shape (T, 2, 2)
        state_cov_by_regime : shape (T, 2, 2, 2)
        state_mean : unconditional filtered mean, shape (T, 2)
        state_var : unconditional filtered variance diagonal, shape (T, 2)
        pair_probs : shape (T, 2, 2)
    """
    y = np.asarray(y, dtype=float)
    T = len(y)

    p11, p22, phi, sigma1, sigma2, g1, g2, mu, tau = theta

    if not (0 < p11 < 1 and 0 < p22 < 1):
        raise ValueError("p11 and p22 must be in (0, 1).")
    if sigma1 <= 0 or sigma2 <= 0 or tau <= 0:
        raise ValueError("sigma1, sigma2, tau must be positive.")

    P = np.array([
        [p11, 1.0 - p11],
        [1.0 - p22, p22]
    ])

    K_reg = 2
    pi = stationary_regime_probs(p11, p22)

    if m_init is None or C_init is None:
        m0, C0 = default_initial_state(theta)
    else:
        m0 = np.asarray(m_init, dtype=float)
        C0 = np.asarray(C_init, dtype=float)

    # Previous filtered regime probabilities
    xi_prev = pi.copy()

    # Previous Gaussian approximation per regime
    m_prev = np.repeat(m0[None, :], K_reg, axis=0)
    C_prev = np.repeat(C0[None, :, :], K_reg, axis=0)

    # Outputs
    loglik = 0.0
    regime_probs = np.zeros((T, K_reg))
    pair_probs = np.zeros((T, K_reg, K_reg))

    state_mean_by_regime = np.zeros((T, K_reg, 2))
    state_cov_by_regime = np.zeros((T, K_reg, 2, 2))

    state_mean = np.zeros((T, 2))
    state_var = np.zeros((T, 2))

    for t in range(T):

        # For each pair i -> j, compute Kalman update
        log_pair_weight = np.zeros((K_reg, K_reg))

        m_pair = np.zeros((K_reg, K_reg, 2))
        C_pair = np.zeros((K_reg, K_reg, 2, 2))

        for i in range(K_reg):
            for j in range(K_reg):

                m_new, C_new, log_pred, _, _ = kalman_predict_update(
                    m_prev=m_prev[i],
                    C_prev=C_prev[i],
                    y_t=y[t],
                    regime_j=j,
                    theta=theta,
                )

                m_pair[i, j] = m_new
                C_pair[i, j] = C_new

                log_pair_weight[i, j] = (
                    np.log(xi_prev[i])
                    + np.log(P[i, j])
                    + log_pred
                )

        # Predictive likelihood
        log_pred_y = logsumexp(log_pair_weight.ravel())
        loglik += log_pred_y

        # Posterior transition-pair probabilities xi_t(i,j)
        pair_prob_t = np.exp(log_pair_weight - log_pred_y)
        pair_probs[t] = pair_prob_t

        # Current regime probabilities xi_t(j)
        xi_curr = pair_prob_t.sum(axis=0)
        regime_probs[t] = xi_curr

        # Collapse mixture for each current regime j
        m_curr = np.zeros((K_reg, 2))
        C_curr = np.zeros((K_reg, 2, 2))

        for j in range(K_reg):
            if xi_curr[j] <= 1e-14:
                # Degenerate fallback
                m_curr[j] = m_pair[0, j]
                C_curr[j] = C_pair[0, j]
                continue

            weights_i_given_j = pair_prob_t[:, j] / xi_curr[j]

            # Collapsed mean
            m_j = np.sum(weights_i_given_j[:, None] * m_pair[:, j, :], axis=0)
            m_curr[j] = m_j

            # Collapsed covariance
            C_j = np.zeros((2, 2))
            for i in range(K_reg):
                diff = m_pair[i, j] - m_j
                C_j += weights_i_given_j[i] * (
                    C_pair[i, j] + np.outer(diff, diff)
                )

            C_curr[j] = 0.5 * (C_j + C_j.T)

        state_mean_by_regime[t] = m_curr
        state_cov_by_regime[t] = C_curr

        # Unconditional filtered state mean and variance
        mean_t = np.sum(xi_curr[:, None] * m_curr, axis=0)
        state_mean[t] = mean_t

        var_diag = np.zeros(2)
        for j in range(K_reg):
            diff = m_curr[j] - mean_t
            var_diag += xi_curr[j] * (np.diag(C_curr[j]) + diff ** 2)

        state_var[t] = var_diag

        # Move forward
        xi_prev = xi_curr
        m_prev = m_curr
        C_prev = C_curr

    return {
        "loglik": loglik,
        "regime_probs": regime_probs,
        "pair_probs": pair_probs,
        "state_mean_by_regime": state_mean_by_regime,
        "state_cov_by_regime": state_cov_by_regime,
        "state_mean": state_mean,
        "state_var": state_var,
        "state_sd": np.sqrt(state_var),
    }

def _kim_neg_loglik(z, y, enforce_label_order=True):
    from rbpf import constrain_theta_rbpf
    try:
        theta = constrain_theta_rbpf(z)
        if enforce_label_order and (theta[4] <= theta[3] or theta[5] <= theta[6]):
            return 1e10
        out = kim_filter_regime_growth(y, theta)
        ll = out["loglik"]
        return -ll if np.isfinite(ll) else 1e10
    except Exception:
        return 1e10


def kim_mle(
    y,
    theta0=None,
    z0=None,
    n_restarts=5,
    noise_scale=0.5,
    compute_se=True,
    se_h=1e-3,
    enforce_label_order=True,
    optimizer_kwargs=None,
    seed=0,
):
    """
    MLE for the 9-parameter regime-switching model via Kim filter log-likelihood.

    Maximizes the approximate log p(y_{1:T} | theta) = sum_t log p_Kim(y_t | y_{1:t-1}, theta)
    by optimizing in unconstrained z-space with L-BFGS-B and multiple random restarts.

    Parameters
    ----------
    y : array-like, shape (T,)
    theta0 : array-like, shape (9,), optional
        Initial theta in constrained space. Defaults to [0.93, 0.85, 0.90, 0.40, 1.80, 0.75, -1.00, 1.00, 0.50].
    z0 : array-like, shape (9,), optional
        Initial z in unconstrained space (takes precedence over theta0).
    n_restarts : int
        Number of optimizer restarts. First uses z0; subsequent ones add N(0, noise_scale) noise.
    noise_scale : float
        Std dev of perturbation applied to z0 for each restart.
    compute_se : bool
        Compute approximate standard errors from finite-difference Hessian + delta method.
    se_h : float
        Step size for Hessian finite differences (default 1e-3).
    enforce_label_order : bool
        Reject configurations with sigma2 <= sigma1 or g1 <= g2 (prevents label switching).
    optimizer_kwargs : dict, optional
        Passed to scipy.optimize.minimize. Defaults to L-BFGS-B with tight tolerances.
    seed : int

    Returns
    -------
    dict with keys:
        success         : bool
        theta_mle       : (9,) MLE in constrained space, or None
        z_mle           : (9,) MLE in unconstrained space, or None
        loglik          : log-likelihood at MLE
        se_theta        : (9,) standard errors in constrained space (delta method), or None
        Sigma_z         : (9,9) covariance matrix in unconstrained space, or None
        hessian_z       : (9,9) numerical Hessian of neg-log-lik in z-space, or None
        optimizer_result: best scipy OptimizeResult
        all_results     : list of all valid OptimizeResult objects
    """
    from rbpf import constrain_theta_rbpf, unconstrain_theta_rbpf
    from scipy.optimize import minimize

    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)

    if optimizer_kwargs is None:
        optimizer_kwargs = {
            "method": "L-BFGS-B",
            "options": {"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-7},
        }

    if z0 is not None:
        z0 = np.asarray(z0, dtype=float)
    else:
        if theta0 is None:
            theta0 = np.array([0.93, 0.85, 0.90, 0.40, 1.80, 0.75, -1.00, 1.00, 0.50])
        z0 = unconstrain_theta_rbpf(np.asarray(theta0, dtype=float))

    best_result = None
    all_results = []

    for k in range(n_restarts):
        z_start = z0.copy() if k == 0 else z0 + rng.normal(0.0, noise_scale, size=9)
        if _kim_neg_loglik(z_start, y, enforce_label_order) >= 1e9:
            continue
        try:
            res = minimize(
                _kim_neg_loglik,
                z_start,
                args=(y, enforce_label_order),
                **optimizer_kwargs,
            )
            if res.fun < 1e9:
                all_results.append(res)
                if best_result is None or res.fun < best_result.fun:
                    best_result = res
        except Exception:
            pass

    if best_result is None:
        return {
            "success": False,
            "theta_mle": None,
            "z_mle": None,
            "loglik": -np.inf,
            "se_theta": None,
            "Sigma_z": None,
            "hessian_z": None,
            "optimizer_result": None,
            "all_results": all_results,
        }

    z_mle = best_result.x
    theta_mle = constrain_theta_rbpf(z_mle)
    loglik = -best_result.fun
    se_theta = None
    Sigma_z = None
    H = None

    if compute_se:
        h = se_h
        n = len(z_mle)
        H = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                zpp = z_mle.copy(); zpp[i] += h; zpp[j] += h
                zpm = z_mle.copy(); zpm[i] += h; zpm[j] -= h
                zmp = z_mle.copy(); zmp[i] -= h; zmp[j] += h
                zmm = z_mle.copy(); zmm[i] -= h; zmm[j] -= h
                H[i, j] = (
                    _kim_neg_loglik(zpp, y) - _kim_neg_loglik(zpm, y)
                    - _kim_neg_loglik(zmp, y) + _kim_neg_loglik(zmm, y)
                ) / (4.0 * h * h)
                H[j, i] = H[i, j]
        try:
            Sigma_z = np.linalg.inv(H)
            p11, p22, phi = theta_mle[0], theta_mle[1], theta_mle[2]
            sigma1, sigma2, tau = theta_mle[3], theta_mle[4], theta_mle[8]
            jac_diag = np.array([
                p11 * (1.0 - p11),
                p22 * (1.0 - p22),
                1.0 - phi ** 2,
                sigma1, sigma2,
                1.0, 1.0, 1.0,
                tau,
            ])
            se_theta = jac_diag * np.sqrt(np.abs(np.diag(Sigma_z)))
        except np.linalg.LinAlgError:
            pass

    return {
        "success": True,
        "theta_mle": theta_mle,
        "z_mle": z_mle,
        "loglik": loglik,
        "se_theta": se_theta,
        "Sigma_z": Sigma_z,
        "hessian_z": H,
        "optimizer_result": best_result,
        "all_results": all_results,
    }


if __name__ == "__main__":
    np.random.seed(42)
    T = 200

    theta_true = np.array([0.93, 0.85, 0.90, 0.40, 1.80, 0.75, -1.00, 1.00, 0.50])
    p11_t, p22_t, phi_t, s1_t, s2_t, g1_t, g2_t, mu_t, tau_t = theta_true

    regimes = np.zeros(T, dtype=int)
    for t in range(1, T):
        if regimes[t - 1] == 0:
            regimes[t] = 0 if np.random.rand() < p11_t else 1
        else:
            regimes[t] = 1 if np.random.rand() < p22_t else 0

    x = np.zeros(T)
    for t in range(1, T):
        sigma_j = s1_t if regimes[t] == 0 else s2_t
        x[t] = phi_t * x[t - 1] + sigma_j * np.random.randn()

    g_j = np.where(regimes == 0, g1_t, g2_t)
    y_sim = g_j + mu_t * (x - np.concatenate([[0.0], x[:-1]])) + tau_t * np.random.randn(T)

    out_kim = kim_filter_regime_growth(y=y_sim, theta=theta_true)
    print("Kim loglik at true theta:", out_kim["loglik"])

    mle = kim_mle(y_sim, n_restarts=5, compute_se=True, seed=0)
    print("MLE success:            ", mle["success"])
    print("MLE loglik:             ", round(mle["loglik"], 4))
    names = ["p11", "p22", "phi", "sigma1", "sigma2", "g1", "g2", "mu", "tau"]
    print(f"{'param':>8}  {'true':>8}  {'mle':>8}  {'se':>8}")
    for name, tv, mv, sv in zip(names, theta_true, mle["theta_mle"], mle["se_theta"]):
        print(f"{name:>8}  {tv:>8.4f}  {mv:>8.4f}  {sv:>8.4f}")
    print("n_restarts completed:   ", len(mle["all_results"]))