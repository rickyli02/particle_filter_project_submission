import numpy as np
from utils import systematic_resample, log_normal_pdf_scalar, logsumexp
from regime_switching import (
    stationary_regime_probs, build_matrices,
    default_initial_state, sigmoid, logit,
)
from kim_filter import kalman_predict_update


# Rao-Blackwellized Particle Filter


def rbpf_regime_growth(
    y,
    theta,
    N_particles=1000,
    seed=0,
    resample_threshold=0.5,
    m_init=None,
    C_init=None,
    store_particles=False,
):
    """
    Rao-Blackwellized particle filter for the model:

        P(s_t = j | s_{t-1} = i) = p_ij

        x_t = phi x_{t-1} + sigma_{s_t} eps_t

        y_t = g^*_{s_t} + mu (x_t - x_{t-1}) + tau eta_t

    Augmented continuous state:
        a_t = [x_t, x_{t-1}]'

    Particles sample only s_t.
    Conditional on each regime path, a Kalman filter integrates out a_t.

    Parameters
    ----------
    y : array-like, shape (T,)
        Observed real GDP growth.

    theta : array-like, shape (9,)
        [p11, p22, phi, sigma1, sigma2, g1, g2, mu, tau]

    N_particles : int
        Number of regime particles.

    Returns
    -------
    out : dict
        loglik : estimated log likelihood
        regime_probs : filtered regime probabilities, shape (T, 2)
        state_mean : filtered mean of a_t, shape (T, 2)
        state_var : filtered covariance diagonals, shape (T, 2)
        ess : effective sample size
        optionally particles_s, particles_m, particles_C, weights
    """
    y = np.asarray(y, dtype=float)
    T = len(y)

    p11, p22, phi, sigma1, sigma2, g1, g2, mu, tau = theta

    if not (0 < p11 < 1 and 0 < p22 < 1):
        raise ValueError("p11 and p22 must be in (0, 1).")
    if sigma1 <= 0 or sigma2 <= 0 or tau <= 0:
        raise ValueError("sigma1, sigma2, tau must be positive.")

    rng = np.random.default_rng(seed)

    P = np.array([
        [p11, 1.0 - p11],
        [1.0 - p22, p22]
    ])

    pi = stationary_regime_probs(p11, p22)

    if m_init is None or C_init is None:
        m0, C0 = default_initial_state(theta)
    else:
        m0 = np.asarray(m_init, dtype=float)
        C0 = np.asarray(C_init, dtype=float)

    # Particle arrays
    s_particles = rng.choice(2, size=N_particles, p=pi)

    m_particles = np.repeat(m0[None, :], N_particles, axis=0)
    C_particles = np.repeat(C0[None, :, :], N_particles, axis=0)

    # Outputs
    loglik = 0.0
    regime_probs = np.zeros((T, 2))
    state_mean = np.zeros((T, 2))
    state_var = np.zeros((T, 2))
    ess = np.zeros(T)

    if store_particles:
        particles_s = np.zeros((T, N_particles), dtype=int)
        particles_m = np.zeros((T, N_particles, 2))
        particles_C = np.zeros((T, N_particles, 2, 2))
        weights_store = np.zeros((T, N_particles))
    else:
        particles_s = None
        particles_m = None
        particles_C = None
        weights_store = None

    for t in range(T):

        # Propagate regimes for t > 0
        if t > 0:
            u = rng.random(N_particles)
            s_new = np.empty(N_particles, dtype=int)

            mask0 = (s_particles == 0)
            s_new[mask0] = (u[mask0] >= P[0, 0]).astype(int)

            mask1 = ~mask0
            s_new[mask1] = (u[mask1] >= P[1, 0]).astype(int)

            s_particles = s_new

        # Kalman predict/update for each particle conditional on s_t
        logw = np.zeros(N_particles)

        for n in range(N_particles):
            m_new, C_new, log_pred, _, _ = kalman_predict_update(
                m_prev=m_particles[n],
                C_prev=C_particles[n],
                y_t=y[t],
                regime_j=s_particles[n],
                theta=theta,
            )

            m_particles[n] = m_new
            C_particles[n] = C_new
            logw[n] = log_pred

        # Likelihood contribution
        log_norm_const = logsumexp(logw) - np.log(N_particles)
        loglik += log_norm_const

        # Normalize weights
        logW = logw - logsumexp(logw)
        weights = np.exp(logW)

        # ESS
        ess[t] = 1.0 / np.sum(weights ** 2)

        # Filtered regime probabilities
        regime_probs[t, 0] = np.sum(weights * (s_particles == 0))
        regime_probs[t, 1] = np.sum(weights * (s_particles == 1))

        # Filtered continuous state mean
        mean_t = np.sum(weights[:, None] * m_particles, axis=0)
        state_mean[t] = mean_t

        # Mixture variance diagonal
        second_moment = np.zeros(2)
        for n in range(N_particles):
            diff = m_particles[n] - mean_t
            second_moment += weights[n] * (np.diag(C_particles[n]) + diff ** 2)

        state_var[t] = second_moment

        if store_particles:
            particles_s[t] = s_particles
            particles_m[t] = m_particles
            particles_C[t] = C_particles
            weights_store[t] = weights

        # Resample
        if ess[t] < resample_threshold * N_particles:
            idx = systematic_resample(weights, rng)
            s_particles = s_particles[idx]
            m_particles = m_particles[idx]
            C_particles = C_particles[idx]
            weights = np.full(N_particles, 1.0 / N_particles)

    return {
        "loglik": loglik,
        "regime_probs": regime_probs,
        "state_mean": state_mean,
        "state_var": state_var,
        "state_sd": np.sqrt(state_var),
        "ess": ess,
        "particles_s": particles_s,
        "particles_m": particles_m,
        "particles_C": particles_C,
        "weights": weights_store,
    }


# ---------------------------------------------------------------------------
# Parameter transforms for RBPF-PMMH (9-parameter model)
# ---------------------------------------------------------------------------
# Unconstrained z:
#   [logit_p11, logit_p22, atanh_phi, log_sigma1, log_sigma2, g1, g2, mu, log_tau]
# Constrained theta:
#   [p11, p22, phi, sigma1, sigma2, g1, g2, mu, tau]


def _log_normal_kernel(x, mean, sd):
    """Log N(mean, sd^2) kernel (normalizer cancels in MH ratios)."""
    return -0.5 * ((x - mean) / sd) ** 2


def constrain_theta_rbpf(z):
    return np.array([
        sigmoid(z[0]),  # p11
        sigmoid(z[1]),  # p22
        np.tanh(z[2]),  # phi
        np.exp(z[3]),   # sigma1
        np.exp(z[4]),   # sigma2
        z[5],           # g1 (unconstrained)
        z[6],           # g2 (unconstrained)
        z[7],           # mu (unconstrained)
        np.exp(z[8]),   # tau
    ])


def unconstrain_theta_rbpf(theta):
    p11, p22, phi, sigma1, sigma2, g1, g2, mu, tau = theta
    return np.array([
        logit(p11),
        logit(p22),
        np.arctanh(phi),
        np.log(sigma1),
        np.log(sigma2),
        g1,
        g2,
        mu,
        np.log(tau),
    ])


def log_prior_rbpf(z, enforce_label_order=True):
    """
    Weakly informative prior on unconstrained z for the RBPF model.

    Enforces sigma2 > sigma1 and g1 > g2 to prevent label switching.
    Prior centers:
        p11 ≈ 0.93, p22 ≈ 0.85, phi ≈ 0.90,
        sigma1 ≈ 0.40, sigma2 ≈ 1.80,
        g1 ≈ 0.75, g2 ≈ -1.00, mu ≈ 1.00, tau ≈ 0.50
    """
    theta = constrain_theta_rbpf(z)
    p11, p22, phi, sigma1, sigma2, g1, g2, mu, tau = theta

    if not np.all(np.isfinite(theta)):
        return -np.inf
    if enforce_label_order and not (sigma2 > sigma1 and g1 > g2):
        return -np.inf

    lp  = _log_normal_kernel(z[0], logit(0.93), 1.0)
    lp += _log_normal_kernel(z[1], logit(0.85), 1.2)
    lp += _log_normal_kernel(z[2], np.arctanh(0.90), 0.5)
    lp += _log_normal_kernel(z[3], np.log(0.40), 0.8)
    lp += _log_normal_kernel(z[4], np.log(1.80), 0.8)
    lp += _log_normal_kernel(z[5], 0.75, 1.0)
    lp += _log_normal_kernel(z[6], -1.00, 1.0)
    lp += _log_normal_kernel(z[7], 1.00, 1.0)
    lp += _log_normal_kernel(z[8], np.log(0.50), 0.8)

    return lp


def rbpf_log_likelihood(y, theta, N_particles=1000, seed=0):
    """RBPF log-likelihood wrapper for PMMH."""
    out = rbpf_regime_growth(
        y=y,
        theta=theta,
        N_particles=N_particles,
        seed=seed,
        resample_threshold=0.5,
        store_particles=False,
    )
    return out["loglik"]


# ---------------------------------------------------------------------------
# PMMH using RBPF likelihood
# ---------------------------------------------------------------------------

def pmmh_rbpf(
    y,
    n_iter=5000,
    N_particles=2000,
    step_sizes=None,
    theta0=None,
    z0=None,
    seed=0,
    enforce_label_order=True,
    verbose=True,
    log_every=500,
):
    """
    Particle Marginal Metropolis-Hastings using the RBPF likelihood for the model:

        P(s_t = j | s_{t-1} = i) = p_ij

        x_t = phi x_{t-1} + sigma_{s_t} eps_t

        y_t = g^*_{s_t} + mu (x_t - x_{t-1}) + tau eta_t

    MCMC runs in unconstrained space z = [logit_p11, logit_p22, atanh_phi,
    log_sigma1, log_sigma2, g1, g2, mu, log_tau] with a Gaussian random-walk
    proposal. The RBPF provides an unbiased estimate of p(y | theta) at each step.

    Parameters
    ----------
    y                  : (T,) observed series
    n_iter             : number of MCMC iterations
    N_particles        : particles per RBPF run
    step_sizes         : (9,) proposal std devs in unconstrained space
    theta0             : initial theta in constrained space (used if z0 is None)
    z0                 : initial z in unconstrained space (takes precedence)
    seed               : random seed
    enforce_label_order: reject proposals with sigma2 <= sigma1 or g1 <= g2
    verbose            : print progress
    log_every          : print frequency

    Returns
    -------
    dict with keys:
        samples_theta : (n_iter, 9) samples in constrained space
        samples_z     : (n_iter, 9) samples in unconstrained space
        log_posts     : (n_iter,) current log-posterior trace
        log_liks      : (n_iter,) current RBPF log-likelihood trace
        accepted      : (n_iter,) boolean accept/reject indicators
        acc_rate      : scalar acceptance rate
    """
    y   = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)

    if step_sizes is None:
        step_sizes = np.array([0.15, 0.20, 0.08, 0.08, 0.08, 0.10, 0.10, 0.10, 0.08])
    step_sizes = np.asarray(step_sizes, dtype=float)

    if z0 is None:
        if theta0 is None:
            theta0 = np.array([0.93, 0.85, 0.90, 0.40, 1.80, 0.75, -1.00, 1.00, 0.50])
        z = unconstrain_theta_rbpf(theta0)
    else:
        z = np.asarray(z0, dtype=float)

    theta    = constrain_theta_rbpf(z)
    lp_prior = log_prior_rbpf(z, enforce_label_order=enforce_label_order)
    if not np.isfinite(lp_prior):
        raise ValueError("Initial theta/z has non-finite prior density.")

    pf_seed = int(rng.integers(0, 2 ** 32 - 1))
    log_lik  = rbpf_log_likelihood(y=y, theta=theta, N_particles=N_particles, seed=pf_seed)
    log_post = log_lik + lp_prior

    samples_theta = np.zeros((n_iter, 9))
    samples_z     = np.zeros((n_iter, 9))
    log_posts     = np.zeros(n_iter)
    log_liks      = np.zeros(n_iter)
    accepted      = np.zeros(n_iter, dtype=bool)
    accepts       = 0

    for it in range(n_iter):
        z_prop     = z + rng.normal(loc=0.0, scale=step_sizes)
        theta_prop = constrain_theta_rbpf(z_prop)

        lp_prior_prop = log_prior_rbpf(z_prop, enforce_label_order=enforce_label_order)

        if np.isfinite(lp_prior_prop):
            pf_seed       = int(rng.integers(0, 2 ** 32 - 1))
            log_lik_prop  = rbpf_log_likelihood(y=y, theta=theta_prop, N_particles=N_particles, seed=pf_seed)
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


if __name__ == "__main__":
    # random test run
    random_regime = np.random.randint(0, 2, size=100)
    measurement_noise = np.random.normal(0, 0.5, size=100)
    real_gdp_growth = np.random.normal(0, 1, size=100) * (random_regime * 1.8 + (1 - random_regime) * 0.4) + measurement_noise

    theta = np.array([
        0.93,  # p11
        0.85,  # p22
        0.90,  # phi
        0.40,  # sigma1
        1.80,  # sigma2
        0.75,  # g1*
        -1.00, # g2*
        1.00,  # mu
        0.50,  # tau
    ])

    out_rbpf = rbpf_regime_growth(
        y=real_gdp_growth,
        theta=theta,
        N_particles=5000,
        seed=123,
        resample_threshold=0.5,
        store_particles=False,
    )

    print("RBPF loglik:", out_rbpf["loglik"])