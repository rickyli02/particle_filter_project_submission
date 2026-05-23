import numpy as np
from utils import logsumexp, log_normal_pdf_scalar, log_normal_pdf
from estimation.resampling_methods import systematic_resample


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

# ============================================================
# State-space model utilities
# ============================================================


def stationary_regime_probs(p11, p22):
    """
    Stationary distribution of the two-state Markov chain with transition matrix

        P = [[p11,   1-p11],
             [1-p22, p22  ]]
    """
    denom = 2.0 - p11 - p22
    return np.array([(1.0 - p22) / denom, (1.0 - p11) / denom])

def build_matrices(phi, sigma_j, mu):
    """
    Build state-space matrices for augmented state

        a_t = [x_t, x_{t-1}]'

    State:
        a_t = F a_{t-1} + u_t

    Observation:
        y_t = g_j + H a_t + tau eta_t
    """
    F = np.array([
        [phi, 0.0],
        [1.0, 0.0]
    ])

    Q = np.array([
        [sigma_j ** 2, 0.0],
        [0.0, 0.0]
    ])

    H = np.array([mu, -mu])

    return F, Q, H

def default_initial_state(theta):
    """
    Initial Gaussian approximation for a_{-1} = [x_{-1}, x_{-2}]'.

    This is a rough stationary initialization using average regime variance.
    """
    p11, p22, phi, sigma1, sigma2, g1, g2, mu, tau = theta

    pi = stationary_regime_probs(p11, p22)
    avg_sigma2 = pi[0] * sigma1 ** 2 + pi[1] * sigma2 ** 2

    if abs(phi) < 1:
        var_x = avg_sigma2 / (1.0 - phi ** 2)
    else:
        var_x = 10.0 * avg_sigma2

    cov_lag = phi * var_x

    m0 = np.array([0.0, 0.0])
    C0 = np.array([
        [var_x, cov_lag],
        [cov_lag, var_x]
    ])

    return m0, C0

# ---------------------------------------------------------------------------
# Regime-switching particle filter
# ---------------------------------------------------------------------------

def bootstrap_pf_regime_switching(
    y,
    theta,
    N_particles=1000,
    seed=0,
    resample_threshold=0.5,
    x0_mean=0.0,
    x0_sd=None,
    store_particles=False,
):
    """
    Bootstrap PF for the two-regime state-space model:

        P(s_t = j | s_{t-1} = i) = p_ij

        x_t = phi * x_{t-1} + sigma_{s_t} * eps_t,   eps_t ~ N(0,1)
        y_t = x_t + tau * nu_t,                       nu_t  ~ N(0,1)

    Parameters
    ----------
    y                  : (T,) observed series
    theta              : [p11, p22, phi, sigma1, sigma2, tau]
    N_particles        : number of particles
    seed               : random seed
    resample_threshold : resample when ESS < threshold * N
    x0_mean            : initial state mean
    x0_sd              : initial state std (None → stationary approximation)
    store_particles    : if True, return full particle arrays

    Returns
    -------
    dict with keys:
        loglik, x_filter_mean, x_filter_var, x_filter_sd,
        regime_probs, ess, particles_x*, particles_s*, weights*
    """
    y = np.asarray(y, dtype=float)
    T = len(y)
    p11, p22, phi, sigma1, sigma2, tau = theta

    if not (0.0 < p11 < 1.0 and 0.0 < p22 < 1.0):
        raise ValueError("p11 and p22 must be in (0, 1).")
    if sigma1 <= 0.0 or sigma2 <= 0.0 or tau <= 0.0:
        raise ValueError("sigma1, sigma2, and tau must be positive.")

    rng    = np.random.default_rng(seed)
    sigmas = np.array([sigma1, sigma2])
    P_mat  = np.array([[p11, 1.0 - p11], [1.0 - p22, p22]])
    pi_s   = stationary_regime_probs(p11, p22)

    if x0_sd is None:
        avg_var = pi_s[0] * sigma1 ** 2 + pi_s[1] * sigma2 ** 2
        x0_sd   = np.sqrt(avg_var / (1.0 - phi ** 2)) if abs(phi) < 1.0 else 10.0 * np.sqrt(avg_var)

    x_filter_mean = np.zeros(T)
    x_filter_var  = np.zeros(T)
    regime_probs  = np.zeros((T, 2))
    ess           = np.zeros(T)

    if store_particles:
        particles_x_store = np.zeros((T, N_particles))
        particles_s_store = np.zeros((T, N_particles), dtype=int)
        weights_store     = np.zeros((T, N_particles))
    else:
        particles_x_store = particles_s_store = weights_store = None

    # --- t = 0 ---
    s_particles = rng.choice(2, size=N_particles, p=pi_s)
    x_particles = rng.normal(x0_mean, x0_sd, size=N_particles)

    logw   = log_normal_pdf(y[0], mean=x_particles, sd=tau)
    loglik = logsumexp(logw) - np.log(N_particles)

    logw_norm = logw - logsumexp(logw)
    weights   = np.exp(logw_norm)

    x_filter_mean[0]    = np.sum(weights * x_particles)
    x_filter_var[0]     = np.sum(weights * (x_particles - x_filter_mean[0]) ** 2)
    regime_probs[0, 0]  = np.sum(weights * (s_particles == 0))
    regime_probs[0, 1]  = np.sum(weights * (s_particles == 1))
    ess[0]              = 1.0 / np.sum(weights ** 2)

    if store_particles:
        particles_x_store[0] = x_particles
        particles_s_store[0] = s_particles
        weights_store[0]     = weights

    if ess[0] < resample_threshold * N_particles:
        anc        = systematic_resample(weights, rng)
        x_particles = x_particles[anc]
        s_particles = s_particles[anc]

    # --- t = 1 ... T-1 ---
    for t in range(1, T):
        u     = rng.random(N_particles)
        s_new = np.empty(N_particles, dtype=int)
        mask0 = (s_particles == 0)
        s_new[mask0]  = (u[mask0]  >= P_mat[0, 0]).astype(int)
        s_new[~mask0] = (u[~mask0] >= P_mat[1, 0]).astype(int)
        s_particles = s_new

        x_particles = phi * x_particles + sigmas[s_particles] * rng.normal(size=N_particles)

        logw   = log_normal_pdf(y[t], mean=x_particles, sd=tau)
        loglik += logsumexp(logw) - np.log(N_particles)

        logw_norm = logw - logsumexp(logw)
        weights   = np.exp(logw_norm)

        x_filter_mean[t]   = np.sum(weights * x_particles)
        x_filter_var[t]    = np.sum(weights * (x_particles - x_filter_mean[t]) ** 2)
        regime_probs[t, 0] = np.sum(weights * (s_particles == 0))
        regime_probs[t, 1] = np.sum(weights * (s_particles == 1))
        ess[t]             = 1.0 / np.sum(weights ** 2)

        if store_particles:
            particles_x_store[t] = x_particles
            particles_s_store[t] = s_particles
            weights_store[t]     = weights

        if ess[t] < resample_threshold * N_particles:
            anc         = systematic_resample(weights, rng)
            x_particles = x_particles[anc]
            s_particles = s_particles[anc]

    return {
        "loglik":         loglik,
        "x_filter_mean":  x_filter_mean,
        "x_filter_var":   x_filter_var,
        "x_filter_sd":    np.sqrt(x_filter_var),
        "regime_probs":   regime_probs,
        "ess":            ess,
        "particles_x":    particles_x_store,
        "particles_s":    particles_s_store,
        "weights":        weights_store,
    }


# ---------------------------------------------------------------------------
# Parameter transforms for PMMH
# ---------------------------------------------------------------------------

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(p):
    p = np.asarray(p)
    return np.log(p / (1.0 - p))


def constrain_theta(z):
    """
    Unconstrained → constrained parameter transform.

        z     = [logit_p11, logit_p22, atanh_phi, log_sigma1, log_sigma2, log_tau]
        theta = [p11,       p22,       phi,       sigma1,     sigma2,     tau    ]
    """
    return np.array([
        sigmoid(z[0]),
        sigmoid(z[1]),
        np.tanh(z[2]),
        np.exp(z[3]),
        np.exp(z[4]),
        np.exp(z[5]),
    ])


def unconstrain_theta(theta):
    """
    Constrained → unconstrained parameter transform (inverse of constrain_theta).
    """
    p11, p22, phi, sigma1, sigma2, tau = theta
    return np.array([
        logit(p11),
        logit(p22),
        np.arctanh(phi),
        np.log(sigma1),
        np.log(sigma2),
        np.log(tau),
    ])


# ---------------------------------------------------------------------------
# Prior and likelihood for PMMH
# ---------------------------------------------------------------------------

def _log_normal_kernel(x, mean, sd):
    """Log N(mean, sd^2) up to additive constant (normalizer cancels in MH)."""
    return -0.5 * ((x - mean) / sd) ** 2


def log_prior_z(z, enforce_sigma_order=True):
    """
    Weakly informative prior on the unconstrained parameter vector z.

        z = [logit_p11, logit_p22, atanh_phi, log_sigma1, log_sigma2, log_tau]

    Priors are centered around:
        p11    ≈ 0.97  (normal regime very persistent)
        p22    ≈ 0.60  (crisis regime moderate persistence)
        phi    ≈ 0.85
        sigma1 ≈ 0.60  (normal-regime shock)
        sigma2 ≈ 3.00  (crisis-regime shock)
        tau    ≈ 0.50

    If enforce_sigma_order=True, sigma2 > sigma1 is enforced to avoid
    label switching between the two regimes.
    """
    theta = constrain_theta(z)
    p11, p22, phi, sigma1, sigma2, tau = theta

    if not np.all(np.isfinite(theta)):
        return -np.inf
    if enforce_sigma_order and not (sigma2 > sigma1):
        return -np.inf

    lp  = _log_normal_kernel(z[0], logit(0.97), 1.0)   # logit p11
    lp += _log_normal_kernel(z[1], logit(0.60), 1.2)   # logit p22
    lp += _log_normal_kernel(z[2], np.arctanh(0.85), 1.0)  # atanh phi
    lp += _log_normal_kernel(z[3], np.log(0.60), 1.0)  # log sigma1
    lp += _log_normal_kernel(z[4], np.log(3.00), 1.0)  # log sigma2
    lp += _log_normal_kernel(z[5], np.log(0.50), 1.0)  # log tau

    return lp


def pf_log_likelihood_regime(y, theta, N_particles=1000, seed=0):
    """Particle filter log-likelihood wrapper for PMMH (regime-switching model)."""
    out = bootstrap_pf_regime_switching(
        y=y,
        theta=theta,
        N_particles=N_particles,
        seed=seed,
        resample_threshold=0.5,
        store_particles=False,
    )
    return out["loglik"]


# ---------------------------------------------------------------------------
# PMMH for the regime-switching SSM
# ---------------------------------------------------------------------------

def pmmh_regime_switching(
    y,
    n_iter=5000,
    N_particles=2000,
    step_sizes=None,
    theta0=None,
    z0=None,
    seed=0,
    enforce_sigma_order=True,
    verbose=True,
    log_every=500,
):
    """
    Particle Marginal Metropolis-Hastings for the regime-switching SSM.

    Model:
        P(s_t = j | s_{t-1} = i) = p_ij
        x_t = phi * x_{t-1} + sigma_{s_t} * eps_t,   eps_t ~ N(0,1)
        y_t = x_t + tau * nu_t,                       nu_t  ~ N(0,1)

    PMMH is run in unconstrained space z = [logit_p11, logit_p22, atanh_phi,
    log_sigma1, log_sigma2, log_tau] with a Gaussian random-walk proposal.

    Parameters
    ----------
    y                   : (T,) observed series
    n_iter              : number of MCMC iterations
    N_particles         : particles per PF run
    step_sizes          : (6,) proposal std devs in unconstrained space
    theta0              : initial theta in constrained space (used if z0 is None)
    z0                  : initial z in unconstrained space (takes precedence)
    seed                : random seed
    enforce_sigma_order : if True, reject proposals with sigma2 <= sigma1
    verbose             : print progress
    log_every           : print frequency

    Returns
    -------
    dict with keys:
        samples_theta : (n_iter, 6) samples in constrained space
        samples_z     : (n_iter, 6) samples in unconstrained space
        log_posts     : (n_iter,) current log-posterior trace
        log_liks      : (n_iter,) current PF log-likelihood trace
        accepted      : (n_iter,) boolean accept/reject indicators
        acc_rate      : scalar acceptance rate
    """
    y   = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)

    if step_sizes is None:
        step_sizes = np.array([0.15, 0.20, 0.08, 0.08, 0.08, 0.08])
    step_sizes = np.asarray(step_sizes, dtype=float)

    if z0 is None:
        if theta0 is None:
            theta0 = np.array([0.97, 0.60, 0.85, 0.60, 3.00, 0.50])
        z = unconstrain_theta(theta0)
    else:
        z = np.asarray(z0, dtype=float)

    theta    = constrain_theta(z)
    lp_prior = log_prior_z(z, enforce_sigma_order=enforce_sigma_order)
    if not np.isfinite(lp_prior):
        raise ValueError("Initial theta/z has non-finite prior density.")

    pf_seed = int(rng.integers(0, 2 ** 32 - 1))
    log_lik = pf_log_likelihood_regime(y=y, theta=theta, N_particles=N_particles, seed=pf_seed)
    log_post = log_lik + lp_prior

    samples_theta = np.zeros((n_iter, 6))
    samples_z     = np.zeros((n_iter, 6))
    log_posts     = np.zeros(n_iter)
    log_liks      = np.zeros(n_iter)
    accepted      = np.zeros(n_iter, dtype=bool)
    accepts       = 0

    for it in range(n_iter):
        z_prop     = z + rng.normal(loc=0.0, scale=step_sizes)
        theta_prop = constrain_theta(z_prop)

        lp_prior_prop = log_prior_z(z_prop, enforce_sigma_order=enforce_sigma_order)

        if np.isfinite(lp_prior_prop):
            pf_seed       = int(rng.integers(0, 2 ** 32 - 1))
            log_lik_prop  = pf_log_likelihood_regime(y=y, theta=theta_prop, N_particles=N_particles, seed=pf_seed)
            log_post_prop = log_lik_prop + lp_prior_prop

            if np.log(rng.uniform()) < log_post_prop - log_post:
                z, theta   = z_prop, theta_prop
                log_lik    = log_lik_prop
                log_post   = log_post_prop
                accepts   += 1
                accepted[it] = True

        samples_theta[it] = theta
        samples_z[it]     = z
        log_posts[it]     = log_post
        log_liks[it]      = log_lik

        if verbose and ((it + 1) % log_every == 0):
            print(
                f"[{it+1:>6}/{n_iter}] "
                f"acc_rate={accepts/(it+1):.3f}  "
                f"loglik={log_lik:.2f}  "
                f"theta={np.round(theta, 3)}"
            )

    return {
        "samples_theta": samples_theta,
        "samples_z":     samples_z,
        "log_posts":     log_posts,
        "log_liks":      log_liks,
        "accepted":      accepted,
        "acc_rate":      accepts / n_iter,
    }
