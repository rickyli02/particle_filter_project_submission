import numpy as np
from utils import log_normal_pdf_scalar, logsumexp, systematic_resample
from models.base import StateSpaceModel

    




def log_likelihood(y, x, alpha, tau, phi, sigma):
    """
    Complete-data log-likelihood log p(x_{1:T}, y_{1:T} | theta).

    Used for MLE of the latent state path given fixed parameters.
    """
    transition_resid  = x[1:] - phi * x[:-1]
    measurement_resid = y - alpha * x

    ll_transition  = -0.5 * np.sum(transition_resid ** 2 / sigma ** 2) - (len(x) - 1) * np.log(sigma)
    ll_measurement = -0.5 * np.sum(measurement_resid ** 2 / tau ** 2) - len(x) * np.log(tau)

    return ll_transition + ll_measurement


def pf_log_likelihood(y, phi, alpha, sigma, tau, N_particles=500):
    """
    Marginal log-likelihood log p(y_{1:T} | theta) via bootstrap PF.

    Per step: log p(y_t | y_{1:t-1}) ≈ log( mean_i N(y_t; alpha*x_t^i, tau^2) )
    """
    T = len(y)
    N = N_particles
    rng = np.random.default_rng()
    particles     = np.random.normal(0, sigma, size=N)
    log_lik_total = 0.0

    for t_step in range(T):
        particles = phi * particles + np.random.normal(0, sigma, size=N)

        residuals = y[t_step] - alpha * particles
        log_w     = -0.5 * (residuals / tau) ** 2 - np.log(tau)

        max_log_w = log_w.max()
        log_lik_total += max_log_w + np.log(np.mean(np.exp(log_w - max_log_w)))

        w  = np.exp(log_w - max_log_w)
        w /= w.sum()
        if 1.0 / np.sum(w ** 2) < N / 2:
            particles = particles[systematic_resample(w, rng)]

    return log_lik_total


def neg_log_lik(params, y, N_particles=1000, n_avg=3):
    """
    Negative marginal log-likelihood for Nelder-Mead optimisation.

    params = [phi, alpha, log_sigma, log_tau] (sigma, tau in log-space for positivity).
    Averages over n_avg PF runs to smooth Monte Carlo noise.
    """
    phi, alpha, log_sigma, log_tau = params
    sigma = np.exp(log_sigma)
    tau   = np.exp(log_tau)

    if abs(phi) >= 1.0 or sigma < 1e-6 or tau < 1e-6:
        return 1e10

    lls = [pf_log_likelihood(y, phi, alpha, sigma, tau, N_particles) for _ in range(n_avg)]
    return -np.mean(lls)


def kalman_negloglike_alpha_fixed(raw_params, y):
    """
    Kalman filter negative log-likelihood with alpha fixed to 1.

    raw_params = [raw_phi, raw_sigma, raw_tau] in unconstrained space:
        phi   = tanh(raw_phi)       — enforces stationarity
        sigma = exp(raw_sigma)
        tau   = exp(raw_tau)
    """
    raw_phi, raw_sigma, raw_tau = raw_params
    phi   = np.tanh(raw_phi)
    alpha = 1.0
    sigma = np.exp(raw_sigma)
    tau   = np.exp(raw_tau)

    m   = 0.0
    P   = sigma ** 2 / (1 - phi ** 2)
    nll = 0.0

    for yt in y:
        y_mean    = alpha * m
        S         = alpha ** 2 * P + tau ** 2
        innovation = yt - y_mean

        nll += 0.5 * (np.log(2 * np.pi) + np.log(S) + innovation ** 2 / S)

        K = P * alpha / S
        m = m + K * innovation
        P = (1 - K * alpha) * P

        m = phi * m
        P = phi ** 2 * P + sigma ** 2

    return nll


def log_prior(phi, alpha, sigma, tau):
    """
    Weakly informative priors for the linear SSM.

        phi   ~ Uniform(-1, 1)
        alpha ~ N(1, 5^2)
        log sigma ~ N(0, 2^2)
        log tau   ~ N(0, 2^2)
    """
    if abs(phi) >= 1 or sigma <= 0 or tau <= 0:
        return -np.inf
    lp  = -0.5 * ((alpha - 1) / 5.0) ** 2
    lp += -0.5 * (np.log(sigma) / 2.0) ** 2
    lp += -0.5 * (np.log(tau)   / 2.0) ** 2
    return lp


def pmmh(y, n_iter=2000, N_particles=300, step_sizes=None, theta0=None, seed=0):
    """
    Particle Marginal Metropolis-Hastings for the linear Gaussian SSM.

    State vector: (phi, alpha, log_sigma, log_tau).
    Proposal: Gaussian random walk with diagonal covariance.

    Returns
    -------
    samples  : (n_iter, 4) array of (phi, alpha, sigma, tau)
    acc_rate : empirical acceptance rate
    """
    rng = np.random.default_rng(seed)

    if step_sizes is None:
        step_sizes = np.array([0.015, 0.05, 0.03, 0.03])
    if theta0 is None:
        theta0 = np.array([0.9, 1.5, np.log(0.5), np.log(1.0)])

    theta = theta0.copy()
    phi, alpha = theta[0], theta[1]
    sigma, tau = np.exp(theta[2]), np.exp(theta[3])

    log_lik  = pf_log_likelihood(y, phi, alpha, sigma, tau, N_particles)
    log_post = log_lik + log_prior(phi, alpha, sigma, tau)

    samples = np.zeros((n_iter, 4))
    accepts = 0

    for i in range(n_iter):
        theta_p = theta + rng.normal(0, step_sizes)
        phi_p, alpha_p = theta_p[0], theta_p[1]
        sigma_p, tau_p = np.exp(theta_p[2]), np.exp(theta_p[3])

        lp_prior_p = log_prior(phi_p, alpha_p, sigma_p, tau_p)
        if not np.isfinite(lp_prior_p):
            samples[i] = [phi, alpha, sigma, tau]
            continue

        log_lik_p  = pf_log_likelihood(y, phi_p, alpha_p, sigma_p, tau_p, N_particles)
        log_post_p = log_lik_p + lp_prior_p

        if np.log(rng.uniform()) < log_post_p - log_post:
            theta, log_post = theta_p, log_post_p
            phi, alpha, sigma, tau = phi_p, alpha_p, sigma_p, tau_p
            accepts += 1

        samples[i] = [phi, alpha, sigma, tau]

        if (i + 1) % 500 == 0:
            print(f"Iteration {i + 1}/{n_iter}, acceptance rate so far: {accepts / (i + 1):.3f}")

    return samples, accepts / n_iter
