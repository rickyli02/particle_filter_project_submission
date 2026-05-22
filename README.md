# Particle Filter Project

Sequential Monte Carlo and Kalman-filter methods for regime-switching state-space models, with an application to macroeconomic data.

---

## Repository layout

```
particle_filter_project/
├── src/
│   ├── particle_filter.py       # Core bootstrap PF implementations
│   ├── state_space_model.py     # SSM abstract class, linear Gaussian model, PMMH
│   ├── regime_switching.py      # Two-regime bootstrap PF + PMMH
│   ├── rbpf.py                  # Rao-Blackwellized PF + PMMH (9-param growth model)
│   ├── kim_filter.py            # Kim filter + MLE for growth model
│   ├── regime_change_macro.py   # Full macro regime-switching model (Kim filter/smoother/forecast)
│   ├── analysis_utils.py        # ARIMA model-selection utilities
│   └── utils.py                 # Shared numerical primitives (logsumexp, softmax, log_normal_pdf_scalar)
├── data/
│   └── synthetic_data.py        # Synthetic data generators
└── notebooks/
    ├── MC_Projectv2.ipynb
    ├── fred_regime_macro.ipynb
    ├── pmmh_blocked_real_gdp.ipynb
    ├── real_data_analysis.ipynb
    └── testing_estimation.ipynb
```

---

## Source files

### `src/particle_filter.py`
Bootstrap particle filters for univariate latent state-space models.

| Symbol | Description |
|--------|-------------|
| `_systematic_resample(weights)` | Systematic resampling using `np.random.uniform` (no explicit RNG — not reproducible) |
| `systematic_resample(weights, rng)` | Same algorithm with an explicit `np.random.Generator` for reproducible seeds |
| `particle_filter(y, phi, alpha, sigma, tau, N)` | Bootstrap PF for AR(1) latent state with Gaussian measurement noise |
| `particle_filter_student_t(...)` | Same PF but with a Student-t measurement likelihood (robust to outliers) |
| `particle_filter_ARMA(...)` | Bootstrap PF for a latent ARMA(1,3) state-space model; returns log-likelihood estimate |

> Also contains two empty stub classes (`ResamplingMethod`, `StateSpaceModel`) that are superseded by `state_space_model.py`.

---

### `src/state_space_model.py`
Abstract base class for SSMs, a concrete linear Gaussian implementation, and inference utilities.

| Symbol | Description |
|--------|-------------|
| `StateSpaceModel` | Abstract base class defining the SSM interface (`transition`, `observation`, `log_*_density`) |
| `SimpleLinearGaussianSSM` | 1-D AR(1) state, 1-D Gaussian observation; implements the full `StateSpaceModel` interface |
| `regime_switching_SSM` | Multi-regime linear Gaussian SSM built on the same abstract class |
| `logsumexp(a)` | Numerically stable log-sum-exp (scalar array) |
| `log_normal_pdf_scalar(y, mean, var)` | Log density of N(mean, var), scalar |
| `log_likelihood(y, x, alpha, tau, phi, sigma)` | Complete-data log-likelihood for the AR(1) SSM |
| `pf_log_likelihood(y, phi, alpha, sigma, tau, N)` | Marginal log-likelihood estimate via bootstrap PF |
| `neg_log_lik(params, y, N, n_avg)` | Negative log-likelihood averaged over `n_avg` PF runs; used with Nelder-Mead |
| `kalman_negloglike_alpha_fixed(raw_params, y)` | Kalman-filter NLL with `alpha=1` fixed and parameters in unconstrained space |
| `log_prior(phi, alpha, sigma, tau)` | Weakly informative prior for the linear SSM |
| `pmmh(y, n_iter, N_particles, ...)` | Particle Marginal Metropolis-Hastings for the linear Gaussian SSM |

---

### `src/regime_switching.py`
Two-regime bootstrap particle filter and PMMH for the model:

```
x_t = phi * x_{t-1} + sigma_{s_t} * eps_t
y_t = x_t + tau * nu_t
```

| Symbol | Description |
|--------|-------------|
| `log_normal_pdf(y, mean, sd)` | Vectorized log N(mean, sd²) — note: takes `sd`, not variance |
| `logsumexp(a)` | Same as in `state_space_model.py` |
| `log_normal_pdf_scalar(y, mean, var)` | Same as in `state_space_model.py` |
| `systematic_resample(weights, rng)` | Same as in `particle_filter.py` |
| `stationary_regime_probs(p11, p22)` | Stationary distribution of the 2-state Markov chain |
| `build_matrices(phi, sigma_j, mu)` | Builds (F, Q, H) for the augmented state `[x_t, x_{t-1}]` |
| `default_initial_state(theta)` | Approximate stationary initial mean and covariance for the augmented state |
| `bootstrap_pf_regime_switching(y, theta, N, ...)` | Bootstrap PF for the two-regime model; returns log-likelihood and filtered quantities |
| `sigmoid`, `logit` | Transforms for probability parameters |
| `constrain_theta(z)`, `unconstrain_theta(theta)` | Bijective transforms between constrained and unconstrained parameter spaces |
| `_log_normal_kernel(x, mean, sd)` | Log-Gaussian kernel (normalizer-free) used in prior evaluation |
| `log_prior_z(z, enforce_sigma_order)` | Weakly informative prior over unconstrained parameters |
| `pf_log_likelihood_regime(y, theta, N, seed)` | PF log-likelihood wrapper for PMMH |
| `pmmh_regime_switching(y, n_iter, N, ...)` | PMMH for the two-regime model in unconstrained space |

---

### `src/rbpf.py`
Rao-Blackwellized particle filter (RBPF) and PMMH for the 9-parameter growth model:

```
x_t  = phi * x_{t-1} + sigma_{s_t} * eps_t
y_t  = g*_{s_t} + mu*(x_t - x_{t-1}) + tau * eta_t
```

Particles represent only the discrete regime sequence; the continuous state is integrated out per-particle via a Kalman filter, reducing Monte Carlo variance.

| Symbol | Description |
|--------|-------------|
| `_log_normal_kernel(x, mean, sd)` | Same as the one in `regime_switching.py` |
| `constrain_theta_rbpf(z)`, `unconstrain_theta_rbpf(theta)` | Parameter transforms for the 9-parameter model (adds `g1`, `g2`, `mu`) |
| `log_prior_rbpf(z, enforce_label_order)` | Prior with label-switching constraints (sigma2>sigma1, g1>g2) |
| `rbpf_regime_growth(y, theta, N, ...)` | Core RBPF; returns log-likelihood, filtered regime probabilities, Rao-Blackwellized state moments |
| `rbpf_log_likelihood(y, theta, N, seed)` | Log-likelihood wrapper for PMMH |
| `pmmh_rbpf(y, n_iter, N, ...)` | PMMH using the RBPF likelihood |

---

### `src/kim_filter.py`
Kim (1994) approximate filter and MLE for the same 9-parameter growth model used in `rbpf.py`.

| Symbol | Description |
|--------|-------------|
| `kalman_predict_update(m_prev, C_prev, y_t, regime_j, theta)` | Single Kalman predict-and-update step conditional on regime `j` |
| `kim_filter_regime_growth(y, theta, ...)` | Kim filter: runs `K²` Kalman branches per time step, then collapses to `K` Gaussians |
| `_kim_neg_loglik(z, y, enforce_label_order)` | Objective for MLE optimization |
| `kim_mle(y, theta0, n_restarts, compute_se, ...)` | MLE via L-BFGS-B with multiple restarts; optionally computes SEs via finite-difference Hessian |

---

### `src/regime_change_macro.py`
Full 4-observable macroeconomic regime-switching model with a 6-dimensional latent state:

```
z_t = [output gap, potential growth, natural unemployment, expected inflation, neutral rate, lagged output gap]
y_t = [GDP growth, unemployment, inflation, nominal rate]
```

Supports K regimes (typically 2–4), covariate-dependent transition probabilities, and the full Kim filter/smoother/forecast pipeline.

| Symbol | Description |
|--------|-------------|
| `ModelDims`, `RegimeStructure`, `MacroParams` | Dataclasses encoding model dimensions, parameter-sharing structure, and structural parameters |
| `row_softmax(X)`, `symmetrize(M)` | Numerically stable softmax; symmetric matrix enforcer |
| `kalman_predict(m, P, A, a, Q)` | Linear Kalman prediction step |
| `kalman_update(m_pred, P_pred, y_t, H, b, R)` | Kalman update with Cholesky-based log-likelihood computation |
| `RegimeMacroModel` | Model class; builds regime-specific matrices (A, Q, H, R, b) and the covariate-dependent transition matrix |
| `KimFilterResult`, `kim_filter(model, y, covariates)` | Kim filter for the full macro model |
| `SimulateResult`, `simulate(model, T, seed)` | Draw synthetic (y, z, s) paths from the model |
| `KimSmootherResult`, `kim_smoother(model, filter_result, ...)` | Backward Kim smoother producing P(s_t \| y_{1:T}) and smoothed state moments |
| `ForecastResult`, `forecast(model, filter_result, y_obs, n_ahead)` | Multi-step forecast with Gaussian mixture propagation and confidence intervals |
| `collapse_gaussian_mixture(means, covs, weights)` | Moment-match a Gaussian mixture to a single Gaussian |
| `logsumexp_2d(X)` | Log-sum-exp over a 2-D array |
| `ParamPacker` | Abstract base for converting optimizer vectors to/from `MacroParams` (must be subclassed) |
| `negative_loglik(theta_raw, packer, y, covariates)` | Objective function for MLE |
| `fit_mle(y, dims, theta0_raw, ...)` | MLE driver using L-BFGS-B |

---

### `src/analysis_utils.py`
Utilities for exploratory time-series analysis.

| Symbol | Description |
|--------|-------------|
| `compare_arima_models(y, p_values, d_values, q_values, ...)` | Fits all `ARIMA(p,d,q)` combinations, computes AIC/BIC/HQIC, and runs a Ljung-Box test; returns a `DataFrame` sorted by BIC |

---

### `src/utils.py`
Shared low-level numerical helpers.

| Symbol | Description |
|--------|-------------|
| `logsumexp(a, axis)` | Numerically stable log-sum-exp with optional `axis` argument |
| `softmax(x)` | Row-wise softmax |
| `log_normal_pdf_scalar(x, mean, var)` | Log density of N(mean, var), scalar |

---

### `data/synthetic_data.py`
Functions for generating ground-truth data used in testing.

| Symbol | Description |
|--------|-------------|
| `generate_synthetic_data(T, A, C, Q, R)` | Simulates a linear Gaussian SSM; validates stationarity of A |
| `generate_regime_switching_data(T, A_list, C_list, Q_list, R_list, P, pi)` | Simulates a Markov-switching SSM with per-regime matrices |

---

## Model hierarchy (top-down)

```
regime_change_macro.py   ← most general macro model (6-state, 4-obs, K regimes)
        │
        ├── kim_filter.py        ← 9-param growth model, Kim filter + MLE
        │        └── rbpf.py     ← 9-param growth model, RBPF + PMMH
        │
        └── regime_switching.py  ← 6-param two-regime model, bootstrap PF + PMMH
                 └── particle_filter.py  ← 4-param AR(1), bootstrap PF
```

`state_space_model.py` defines the SSM abstract interface used by the simpler models; `utils.py` provides shared numerical primitives that should be imported across all other files.
