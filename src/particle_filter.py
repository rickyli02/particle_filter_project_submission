import numpy as np
import scipy.stats as stats


def _systematic_resample(weights):
    """Systematic resampling — lower variance than multinomial."""
    N = len(weights)
    positions = (np.arange(N) + np.random.uniform()) / N
    cumsum = np.cumsum(weights)
    indices = np.searchsorted(cumsum, positions)
    return indices


def systematic_resample(weights, rng):
    """Systematic resampling with an explicit RNG (for reproducible seeds)."""
    N = len(weights)
    positions = (rng.random() + np.arange(N)) / N
    cumsum = np.cumsum(weights)
    indices = np.zeros(N, dtype=int)
    i = j = 0
    while i < N:
        if positions[i] < cumsum[j]:
            indices[i] = j
            i += 1
        else:
            j += 1
    return indices


def particle_filter(y, phi, alpha, sigma, tau, N_particles=10000):
    """
    Bootstrap particle filter for the linear Gaussian state-space model.

        x_t = phi * x_{t-1} + eps_t,   eps_t ~ N(0, sigma^2)
        y_t = alpha * x_t + nu_t,       nu_t  ~ N(0, tau^2)

    Parameters
    ----------
    y           : (T,) observed data
    phi, alpha  : state / measurement parameters
    sigma, tau  : process / measurement noise std devs
    N_particles : number of particles

    Returns
    -------
    x_est        : (T,) filtered state estimates (weighted mean per step)
    x_particles  : (T, N) particle trajectories
    weights_hist : (T, N) normalised weights at each step
    """
    T = len(y)
    N = N_particles
    particles = np.random.normal(0, sigma, size=N)
    weights   = np.ones(N) / N
    x_est        = np.zeros(T)
    x_particles  = np.zeros((T, N))
    weights_hist = np.zeros((T, N))

    for t_step in range(T):
        particles = phi * particles + np.random.normal(0, sigma, size=N)

        log_lik = -0.5 * ((y[t_step] - alpha * particles) / tau) ** 2
        log_lik -= log_lik.max()
        weights = np.exp(log_lik)
        weights /= weights.sum()

        x_est[t_step] = np.dot(weights, particles)
        x_particles[t_step] = particles
        weights_hist[t_step] = weights

        n_eff = 1.0 / np.sum(weights ** 2)
        if n_eff < N / 2:
            indices = _systematic_resample(weights)
            particles = particles[indices]
            weights = np.ones(N) / N

    return x_est, x_particles, weights_hist


def particle_filter_student_t(y, phi, alpha, sigma, scale, df, N_particles=10000):
    """
    Bootstrap PF with Student-t measurement likelihood — robust to outliers.

        x_t = phi * x_{t-1} + eps_t,   eps_t ~ N(0, sigma^2)
        y_t = alpha * x_t + nu_t,       nu_t  ~ t(df, scale)

    Parameters
    ----------
    y           : (T,) observed data
    phi, alpha  : state / measurement parameters
    sigma       : process noise std dev
    scale       : Student-t scale parameter for measurement noise
    df          : Student-t degrees of freedom
    N_particles : number of particles

    Returns
    -------
    x_est       : (T,) filtered state estimates
    x_particles : (T, N) particle trajectories
    """
    T = len(y)
    N = N_particles
    particles = np.random.normal(0, sigma, size=N)
    x_est       = np.zeros(T)
    x_particles = np.zeros((T, N))

    for t_step in range(T):
        particles = phi * particles + np.random.normal(0, sigma, size=N)
        residuals = y[t_step] - alpha * particles
        log_lik = stats.t.logpdf(residuals, df=df, scale=scale)
        log_lik -= log_lik.max()
        w = np.exp(log_lik)
        w /= w.sum()
        x_est[t_step] = np.dot(w, particles)
        x_particles[t_step] = particles
        if 1.0 / np.sum(w ** 2) < N / 2:
            particles = particles[_systematic_resample(w)]

    return x_est, x_particles

def particle_filter_ARMA(
    y,
    phi,
    alpha,
    c,
    theta_1,
    theta_2,
    theta_3,
    sigma,
    tau,
    N_particles=10000,
    resample_threshold=0.5,
    seed=None,
):
    """
    Bootstrap particle filter for latent ARMA(1,3) state-space model.

    Latent process:

        x_t = c + phi x_{t-1}
              + nu_t
              + theta_1 nu_{t-1}
              + theta_2 nu_{t-2}
              + theta_3 nu_{t-3},

        nu_t ~ N(0, sigma^2).

    Observation equation:

        y_t = alpha x_t + eps_t,
        eps_t ~ N(0, tau^2).

    Markov state:

        s_t = [x_t, nu_t, nu_{t-1}, nu_{t-2}]'.

    Parameters
    ----------
    y : array-like, shape (T,)
        Observed data.
    phi : float
        AR coefficient. Must satisfy |phi| < 1.
    alpha : float
        Measurement loading.
    c : float
        Latent-state intercept.
    theta_1, theta_2, theta_3 : float
        MA coefficients.
    sigma : float
        Process noise standard deviation.
    tau : float
        Measurement noise standard deviation.
    N_particles : int
        Number of particles.
    resample_threshold : float
        Resample when ESS < resample_threshold * N_particles.
    seed : int or None
        Random seed.

    Returns
    -------
    x_est : ndarray, shape (T,)
        Filtered posterior mean estimate of x_t.
    x_particles_hist : ndarray, shape (T, N)
        Filtered particle cloud for x_t at each time.
    weights_hist : ndarray, shape (T, N)
        Normalized weights at each time before possible resampling.
    loglik : float
        Particle-filter estimate of log likelihood.
    """
    y = np.asarray(y)
    T = len(y)
    N = N_particles

    if abs(phi) >= 1:
        raise ValueError("AR coefficient phi must satisfy |phi| < 1.")
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if tau <= 0:
        raise ValueError("tau must be positive.")
    if alpha == 0:
        raise ValueError("alpha should be nonzero; otherwise x_t is not observed.")

    rng = np.random.default_rng(seed)

    # State transition in column-vector convention:
    #
    # s_t = A s_{t-1} + intercept + b nu_t
    #
    # with s_t = [x_t, nu_t, nu_{t-1}, nu_{t-2}]'.
    A = np.array(
        [
            [phi, theta_1, theta_2, theta_3],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )

    intercept = np.array([c, 0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 0.0, 0.0])

    # Initialize with stationary distribution
    from scipy.linalg import solve_discrete_lyapunov

    m = np.linalg.solve(np.eye(4) - A, intercept)
    Q = sigma**2 * np.outer(b, b)
    Sigma = solve_discrete_lyapunov(A, Q)

    particles = rng.multivariate_normal(mean=m, cov=Sigma, size=N)

    weights = np.ones(N) / N

    x_est = np.zeros(T)
    x_particles_hist = np.zeros((T, N))
    weights_hist = np.zeros((T, N))

    loglik = 0.0

    for t in range(T):
        # Propagate particles.
        nu_t = rng.normal(0.0, sigma, size=N)

        # Since particles are row vectors, use A.T.
        particles = particles @ A.T + intercept + nu_t[:, None] * b

        # Observation prediction.
        y_pred = alpha * particles[:, 0]

        # Full Gaussian log likelihood contribution for each particle.
        log_w_unnorm = (
            -0.5 * np.log(2.0 * np.pi)
            - np.log(tau)
            - 0.5 * ((y[t] - y_pred) / tau) ** 2
        )

        # Stable normalization.
        max_log_w = np.max(log_w_unnorm)
        w_unnorm = np.exp(log_w_unnorm - max_log_w)
        mean_w = np.mean(w_unnorm)

        # Incremental log likelihood estimate.
        loglik += max_log_w + np.log(mean_w)

        weights = w_unnorm / np.sum(w_unnorm)

        # Filtered estimate before resampling.
        x_est[t] = np.sum(weights * particles[:, 0])
        x_particles_hist[t] = particles[:, 0]
        weights_hist[t] = weights

        # Resample if needed.
        ess = 1.0 / np.sum(weights**2)
        if ess < resample_threshold * N:
            indices = systematic_resample(weights, rng=rng)
            particles = particles[indices]
            weights = np.ones(N) / N

    return x_est, x_particles_hist, weights_hist, loglik
