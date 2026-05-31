# Particle Filter Project

Sequential Monte Carlo and Kalman-filter methods for state-space models, with an application to macroeconomic regime-switching.

---

## Architecture

The project is organized as two layers:

- **`src/models/`** — state-space model definitions (dynamics, densities, parameter transforms)
- **`src/estimation/`** — inference algorithms that operate on any compatible model

Models and estimators communicate through the `StateSpaceModel` interface defined in `src/models/base.py`. `src/utils.py` provides shared numerical primitives used across both layers.

---

## Directory structure

```
particle_filter_project/
├── src/
│   ├── models/
│   │   ├── base.py                    # Abstract StateSpaceModel base class
│   │   ├── linear_gaussian.py         # SimpleLinearGaussianSSM, LinearGaussianSSM, FreeLinearGaussianSSM
│   │   ├── linear_gaussian_notes.md   # Kalman filter derivations, score, HMC parameterization
│   │   ├── linear_t.py                # LinearTSSM (t-distributed process noise)
│   │   ├── linear_ARMA.py             # LinearARMASSM (ARMA(1,3) latent state)
│   │   ├── linear_factor.py           # MultivariateObservationLGSSM (single-factor model)
│   │   ├── linear_macro.py            # LinearMacroSSM (HLW-style macro model)
│   │   ├── regime_switching.py        # RegimeSwitchingSSM (general K-regime)
│   │   ├── regime_switching_base.py   # RegimeSwitchingBase, RegimeSwitchingDims, RegimeSwitchingStructure
│   │   ├── regime_switching_macro.py  # RegimeSwitchingMacro (6-state macro model)
│   │   ├── regime_switching_growth.py # RegimeSwitchingGrowth (2-regime output gap model)
│   │   └── regime_switching_complex.py# State-dependent transition probabilities (stub)
│   ├── estimation/
│   │   ├── mcmc.py                    # MCMCBase abstract class (shared MCMC infrastructure)
│   │   ├── resampling_methods.py      # Resampling schemes for the particle filter
│   │   ├── particle_filter.py         # Bootstrap particle filter
│   │   ├── kalman_filter.py           # Kalman filter + RTS smoother
│   │   ├── kim_filter.py              # Kim (1994) approximate filter + smoother
│   │   ├── rbpf.py                    # Rao-Blackwellized particle filter
│   │   ├── pmmh.py                    # PMMH and BlockPMMH
│   │   ├── metropolis_hastings.py     # MetropolisHastings, BlockMetropolisHastings
│   │   ├── hamilton_mc.py             # HamiltonianMC (gradient-based MCMC)
│   │   ├── mle_estimator.py           # MLE via Kalman likelihood
│   │   ├── kde.py                     # Weighted KDE for PF posteriors and MCMC chains
│   │   ├── mala.py                    # MALA (stub)
│   │   └── map_smoothing.py           # MAP smoothing over latent states (stub)
│   ├── utils.py                       # Shared numerical utilities
│   ├── time_series_analysis.py        # ARIMA grid search and BIC comparison
│   └── older_code/                    # Legacy implementations (superseded)
├── notebooks/
│   ├── MC_Projectv3.ipynb             # Main project notebook (filtering, smoothing, prediction, parameter estimation)
│   ├── testing_estimation.ipynb       # Filter comparisons, N-particle sweep, resampling comparison
│   ├── parameter_estimation.ipynb     # MLE, MH, PMMH posterior inference
│   ├── hmc_estimation.ipynb           # HMC vs MH vs MLE comparison
│   ├── basic_results.ipynb            # Empirical study: filtering, MLE, MH, identifiability, RMSE (N=100 trials)
│   ├── macro_data_linear.ipynb        # LG-SSMs on US macro FRED data (simple, factor, HLW, free)
│   └── older_notebooks/               # Earlier notebooks (superseded)
├── data/                              # Data acquisition scripts and cached CSVs
└── results/                           # Output files from multi-trial notebook runs (CSV checkpoints, etc.)
```

---

## Models — `src/models/`

### `base.py`

Abstract base class for all state-space models.

| Symbol | Description |
|--------|-------------|
| `StateSpaceModel` | Base class; defines the full SSM interface |
| `.params` | Property returning `tuple(params_dict.values())` |
| `.generate_data(T)` | Simulate `(states, observations, log_likelihood)` |
| `.sample_initial_distribution()` | Sample `x_0`; should use stationary distribution where possible |
| `.initial_density(x)` | Density `p(x_0 = x)` |
| `.log_initial_density(x)` | Default: `log(initial_density(x))`; override with closed form |
| `.transition(x_prev)` | Sample `x_t \| x_{t-1}` |
| `.observation(x)` | Sample `y_t \| x_t` |
| `.log_transition_density(x_next, x_prev)` | Log `p(x_t \| x_{t-1})` |
| `.log_observation_density(y, x)` | Log `p(y_t \| x_t)` |
| `.update_params(constrained_params)` | Update model attributes in-place; called by PMMH each iteration |
| `.clear_state()` | Reset accumulated mutable runtime state; default no-op |
| `.constrain_params(theta_unc)` | Map unconstrained vector → valid parameter object |
| `.unconstrain_params(constrained_params)` | Inverse; returns flat `np.ndarray` |
| `.describe()` | Human-readable model summary with equations |

Note: MCMC saplers in unconstrained parameter space may suffer from slower mixing when true parameters are close to boundary.

---

### `linear_gaussian.py`

| Class | Model |
|-------|-------|
| `SimpleLinearGaussianSSM(phi, alpha, sigma2, tau2)` | `x_t = φ x_{t-1} + ε_t`, `y_t = α x_t + ν_t`; 1-D latent, 1-D observation, Gaussian noise; parameters are **variances** `σ²`, `τ²` |
| `LinearGaussianSSM(a, c, q, r, b, d, mu_0, p_0)` | `x_t = A x_{t-1} + b + ε_t`, `y_t = C x_t + d + ν_t`; general multivariate; initial distribution defaults to stationary via `solve_discrete_lyapunov` |
| `FreeLinearGaussianSSM(n_latent, n_obs)` | Fully unconstrained multivariate LG-SSM; A lower-triangular (tanh diagonal for stability), Q via Cholesky, R diagonal log-variance; all entries free for MLE |

Both `SimpleLinearGaussianSSM` and `LinearGaussianSSM` implement the full `StateSpaceModel` interface including `update_params`, `constrain_params` / `unconstrain_params`, and `log_likelihood(data)` (Kalman filter recursion, exact marginal likelihood).

`SimpleLinearGaussianSSM` additionally implements:

| Method | Description |
|--------|-------------|
| `.score(data)` | Analytic gradient `∇_θ log p(y\|θ)` w.r.t. `(φ, α, σ², τ²)`; propagates sensitivities through the Kalman recursion in a single O(T) forward pass |
| `.jacobian_constrain_params(u)` | Diagonal Jacobian `dθ/du` of the constrain transform; used by HMC for the chain rule `∇_u ℓ = diag(J) ⊙ ∇_θ ℓ` |

---

### `linear_t.py`

| Class | Model |
|-------|-------|
| `LinearTSSM(alpha, tau, phi, sigma, df)` | Same observation equation as `SimpleLinearGaussianSSM` but process noise is Student-t; useful for testing filter robustness to heavy tails. Implements `constrain_params` / `unconstrain_params` (tanh for `phi`, log for `tau`, `sigma`, `df`) |

---

### `linear_ARMA.py`

| Class | Model |
|-------|-------|
| `LinearARMASSM(phi, alpha, c, theta_1, theta_2, theta_3, sigma, tau)` | ARMA(1,3) latent process lifted to Markov state `s_t = [x_t, ν_t, ν_{t-1}, ν_{t-2}]`; 4-D latent, 1-D observation |

Overrides `clear_state()` to reset `s` and `s_history` between particle filter runs.

---

### `linear_factor.py`

| Class | Model |
|-------|-------|
| `MultivariateObservationLGSSM(phi, sigma2, alphas, tau2s, mus)` | Dynamic single-factor model: 1D latent AR(1) state drives K observations; `y_t^(k) = μ^(k) + α^(k) x_t + ν_t^(k)`; identification via `α^(1) ≡ 1`. Free parameters: `φ`, `σ²`, `α_2, …, α_K`, `τ²_1, …, τ²_K`, `μ_1, …, μ_K` |

---

### `linear_macro.py`

| Class | Model |
|-------|-------|
| `LinearMacroSSM(phi_1, phi_2, lambda_r, c_g, alpha_pi, beta_pi, gamma, rho_i, psi_pi, psi_x, sigma_*, pi_star, ...)` | HLW-style macro state-space model; 8D latent state `[x_t, x_{t-1}, g_t*, ζ_t, r_t*, r*_{t-1}, u_t*, u*_{t-1}]`; 4 observables (GDP growth, inflation, unemployment, nominal rate); time-varying observation intercepts depending on lagged nominal rate |

The model implements custom `.filter()` and `.smoother()` methods (the generic `KalmanFilter` class is not compatible due to time-varying offsets). Three weakly-identified parameters (`σ_g`, `σ_ζ`, `c_g`) are typically fixed at calibrated values during MLE.

---

### `regime_switching.py`

| Class | Model |
|-------|-------|
| `RegimeSwitchingSSM(A_list, C_list, Q_list, R_list, regime_transition_matrix)` | General K-regime Markov-switching linear Gaussian SSM; per-regime matrices `A_k, C_k, Q_k, R_k`; initial regime drawn from stationary distribution of the Markov chain |

---

### `regime_switching_base.py`

Base class for Markov-switching linear Gaussian SSMs. Exposes the interface required by `KimFilter` and `RaoBlackwellizedParticleFilter`.

| Symbol | Description |
|--------|-------------|
| `RegimeSwitchingDims` | Dataclass: `n_regimes`, `state_dim`, `obs_dim` |
| `RegimeSwitchingStructure` | Dataclass: flags controlling which matrices vary by regime (`regime_specific_A/C/Q/R`, `has_state_intercept`, `has_obs_intercept`) |
| `RegimeSwitchingBase` | Base class; stores `A_list, C_list, Q_list, R_list, b_list, d_list`, `regime_transition_matrix`, `regime_probabilities_stationary`; implements `log_likelihood_given_regimes(y, regime_path)` via per-particle Kalman filter |

---

### `regime_switching_macro.py`

Full macroeconomic regime-switching model with a 6-dimensional augmented latent state and 4 observables.

```
State:    z_t = [x_t, g_t*, u_t*, π_t^e, r_t*, x_{t-1}]
Observed: y_t = [GDP growth, unemployment, inflation, nominal rate]

Transition:  z_t = A z_{t-1} + a + ε_t,   ε_t ~ N(0, Q_{s_t})
Observation: y_t = H z_t + b_t(i_{t-1}) + η_t,   η_t ~ N(0, R)
```

Q is regime-specific; A, H, R are shared. The observation intercept `b_t` depends on the lagged nominal rate.

| Symbol | Description |
|--------|-------------|
| `ModelDims` | Dataclass: `n_regimes`, `n_state=6`, `n_obs=4`, `n_covariates` |
| `RegimeStructure` | Dataclass: flags controlling which matrices are regime-specific |
| `MacroParams` | Dataclass: all structural parameters (persistence, long-run means, slopes, Taylor rule, sigmas, transition intercepts) |
| `RegimeSwitchingMacro` | Model class; matrix builders `build_A`, `build_Q`, `build_H`, `build_b`, `build_R`; `transition_probs` for (optionally covariate-dependent) regime transitions; `constrain_params` / `unconstrain_params` using tanh/exp transforms |

---

### `regime_switching_growth.py`

| Class | Model |
|-------|-------|
| `RegimeSwitchingGrowth(p11, p22, phi, sigma1, sigma2, g1, g2, mu, tau)` | Two-regime output-gap model; latent gap `x_t = φ x_{t-1} + σ_{s_t} ε_t`; observed GDP growth `y_t = g*_{s_t} + μ(x_t − x_{t-1}) + τ η_t`; augmented state `[x_t, x_{t-1}]`; inherits from `RegimeSwitchingBase` |

Regime 0 is expansion (low volatility, higher `g*`); regime 1 is contraction.

---

## Estimation — `src/estimation/`

### `resampling_methods.py`

| Symbol | Description |
|--------|-------------|
| `ResamplingMethod` | Abstract base; sets `resample_threshold = 0.5` |
| `SystematicResampling` | Single uniform draw, O(N) — recommended default |
| `StratifiedResampling` | One uniform draw per stratum |
| `ResidualResampling` | Deterministic integer copies + multinomial residual |
| `MultinomialResampling` | N independent draws; high variance, included as baseline |
| `systematic_resample(weights, rng)` | Standalone function returning resampled indices; used internally by models |

---

### `particle_filter.py`

| Symbol | Description |
|--------|-------------|
| `ParticleFilter(model, N_particles, data, resample_method, seed)` | Bootstrap particle filter for any `StateSpaceModel` |
| `.run_filter()` | Returns `(latent_state_estimate, particle_history, weight_history, resample_history, loglik)` |
| `.ESS` | Property: effective sample size `1 / Σ w_i²` |

Resamples when `ESS < resample_threshold × N`. Log-likelihood is accumulated as `Σ_t [logsumexp(log_w_t) − log N]`.

---

### `kalman_filter.py`

Exact inference for `LinearGaussianSSM` only.

| Symbol | Description |
|--------|-------------|
| `KalmanFilter(model, data)` | Kalman filter + RTS smoother |
| `.run_filter()` | Returns `(filtered_means, filtered_covs, loglik)`; stores `predicted_means/covs`, `innovations`, `innovation_covs` |
| `.run_smoother()` | RTS backward pass; returns `(smoothed_means, smoothed_covs)`; must call `run_filter()` first |

Uses Cholesky factorization of the innovation covariance for numerical stability; applies the Joseph-form covariance update to guarantee symmetry and PSD at each step.

---

### `kim_filter.py`

Kim (1994) approximate filter and smoother for Markov-switching linear Gaussian SSMs. Reduces Monte Carlo variance by marginalizing the continuous state analytically via per-regime Kalman filters.

| Symbol | Description |
|--------|-------------|
| `KimFilter(model, data)` | Must expose `n_regimes`, `regime_transition_matrix`, `regime_probabilities_stationary`, `A_list`, `C_list`, `Q_list`, `R_list` |
| `.run_filter()` | Returns `(filtered_means, filtered_probs, loglik)`; Kim collapsing approximation marginalises `s_{t-1}` at each step |
| `.run_smoother()` | Backward pass returning `(smoothed_means, smoothed_probs)`; must call `run_filter()` first |

---

### `rbpf.py`

Rao-Blackwellized Particle Filter for Markov-switching linear Gaussian SSMs. Particles track only the discrete regime sequence; the continuous latent state is marginalized analytically via a per-particle Kalman filter. Inherits from `ParticleFilter`.

| Symbol | Description |
|--------|-------------|
| `RaoBlackwellizedParticleFilter(model, N_particles, data, resample_method, seed)` | Compatible with any model exposing the `RegimeSwitchingBase` interface |
| `.run_filter()` | Returns `(state_estimate, regime_history, weight_history, resample_history, loglik)` |

---

### `mcmc.py`

Abstract base class for all MCMC samplers. Handles the constrain/unconstrain bookkeeping, prior evaluation, and Jacobian correction in one place so subclasses only implement the proposal mechanism.

| Symbol | Description |
|--------|-------------|
| `MCMCBase` | Abstract base; stores `model`, `data`, `theta0`, `step_sizes`, `log_prior`, `prior_space`, `include_jacobian` |
| `._evaluate_loglik(theta_unc)` | Abstract; subclasses return `log p(y\|θ(u))` or an unbiased estimate |
| `.run()` | Abstract; subclasses populate `chain`, `loglik_chain`, `accepted`, `accept_rate` |
| `._log_prior_term(theta_unc)` | Prior contribution including optional `log\|det J\|` correction |
| `._log_abs_det_jacobian(theta_unc)` | `log\|det dθ/du\|` via `model.jacobian_constrain_params`; supports diagonal and general Jacobians |
| `.constrained_chain` | Property: maps every row of the post-run chain through `constrain_params` |
| `.summary(burn)` | Prints posterior mean, std, and acceptance rate for each constrained parameter |

The target density is `π(u) ∝ p(y\|θ(u)) · p_prior(θ(u)) · \|det J(u)\|` for constrained priors, or `π(u) ∝ p(y\|θ(u)) · p_prior(u)` for unconstrained priors.

---

### `metropolis_hastings.py`

Random-walk Metropolis-Hastings with a closed-form log-likelihood (no particle filter).

| Symbol | Description |
|--------|-------------|
| `MetropolisHastings(model, data, n_iter, step_sizes, theta0, log_prior, prior_space, seed)` | Gaussian random-walk proposals in unconstrained space; exact posterior for tractable models |
| `BlockMetropolisHastings(..., blocks)` | Cycles through parameter index groups; each block accepted/rejected independently |

---

### `pmmh.py`

Particle Marginal Metropolis-Hastings. Requires the model to implement `constrain_params`, `unconstrain_params`, and `update_params`.

| Symbol | Description |
|--------|-------------|
| `PMMH(model, particle_filter, n_iter, step_sizes, theta0, log_prior, seed)` | Standard PMMH; Gaussian random walk in unconstrained space |
| `.run()` | Returns `(chain, loglik_chain, accepted)`; `chain` has shape `(n_iter+1, d)` |
| `BlockPMMH(..., blocks)` | Block-update PMMH; cycles through parameter index groups each iteration; each block is accepted/rejected independently |

`theta0` must be in unconstrained space. Use `model.unconstrain_params(...)` to obtain the initial vector. The particle filter's history is cleared automatically before each likelihood evaluation.

---

### `hamilton_mc.py`

Hamiltonian Monte Carlo for models with a closed-form log-likelihood and analytic score. Requires `model.score(data)` and `model.jacobian_constrain_params(u)` in addition to the standard `MCMCBase` interface.

| Symbol | Description |
|--------|-------------|
| `HamiltonianMC(model, data, n_iter, step_size, n_leapfrog, mass_diag, theta0, log_prior, prior_space, seed)` | Gradient-guided proposals via leapfrog integration |
| `._grad_log_target(u)` | Gradient of `log π(u)`: exact for the likelihood term via `model.score` + chain rule `∇_u ℓ = diag(J) ⊙ ∇_θ ℓ`; central finite differences for the cheap prior + Jacobian correction |
| `._leapfrog(u, p, grad_u)` | `n_leapfrog` leapfrog steps of size `step_size`; half-kick → (drift → full-kick) × (L−1) → drift → half-kick |
| `.run()` | Returns `(chain, loglik_chain, accepted)`; each iteration costs `L + 2` Kalman passes |

`mass_diag` (default: ones) is the diagonal of the mass matrix M; tuning it to approximate posterior variances improves mixing. Target acceptance rate is typically 60–80%.

---

### `mle_estimator.py`

Maximum likelihood estimation for state-space models with a tractable log-likelihood.

| Symbol | Description |
|--------|-------------|
| `MLEEstimator(model, data, method, n_restarts, restart_std, seed)` | Maximizes `model.log_likelihood(data)` in unconstrained parameter space via `scipy.optimize.minimize` (default `L-BFGS-B`); supports random restarts |
| `.fit(theta0, fixed_params)` | Run optimization; `theta0` defaults to `model.unconstrain_params(model.params)`. `fixed_params` holds named constrained parameters constant throughout. Returns `MLEResult` |
| `.compute_std_errors(eps)` | Numerical Hessian at the MLE → delta method to return standard errors in the *constrained* space |
| `MLEResult` | Dataclass: `constrained_params`, `unconstrained_params`, `loglik`, `success`, `n_evals`, `message`, `std_errors`; `.summary()` prints a formatted parameter table |

Requires the model to implement `log_likelihood(data)`, `constrain_params`, `unconstrain_params`, and `update_params`. Raises `ValueError` for models without `log_likelihood`.

---

### `kde.py`

Weighted kernel density estimation for particle filter posteriors and MCMC chain marginals.

| Symbol | Description |
|--------|-------------|
| `KDE(particles, weights)` | Gaussian KDE with Silverman bandwidth; `weights=None` uses uniform weights |
| `kde(x_grid)` | Evaluate density on a grid |
| `kde.log_evaluate(x)` | Log density (numerically stable) |
| `kde.sample(n)` | Draw samples from the KDE |
| `particle_posterior_kde(pf, t)` | Extract KDE from a `ParticleFilter` at time step `t` |
| `chain_marginal_kdes(chain, names)` | Return a list of KDEs, one per parameter column of an MCMC chain |

---

## Utilities — `src/utils.py`

| Symbol | Description |
|--------|-------------|
| `logsumexp(a, axis)` | Numerically stable `log Σ exp(a)` with optional axis |
| `softmax(x)` | Row-wise softmax (not numerically stabilized) |
| `row_softmax(x)` | Numerically stable row-wise softmax; used for regime transition probabilities |
| `symmetrize(m)` | Returns `(m + m.T) / 2`; used to enforce covariance symmetry |
| `log_normal_pdf_scalar(y, mean, var)` | Log density of `N(mean, var)` for scalars |
| `log_normal_pdf(y, mean, sd)` | Vectorized log density; takes `sd`, not variance |

---

## Time-series utilities — `src/time_series_analysis.py`

| Symbol | Description |
|--------|-------------|
| `compare_arima_models(y, p_values, d_values, q_values, ...)` | Fit multiple ARIMA(p,d,q) models and compare by AIC, BIC, HQIC, log-likelihood, and residual Ljung-Box p-value; returns a `pd.DataFrame` sorted by BIC |

---

## Notebooks

| Notebook | Contents |
|----------|----------|
| `MC_Projectv3.ipynb` | Main project notebook (Ricky Li & Margie Bold): filtering, fixed-lag and complete smoothing, prediction, likelihood evaluation, and parameter estimation tasks for the linear Gaussian SSM |
| `testing_estimation.ipynb` | Single-run particle filter; Monte Carlo RMSE; effect of particle count on RMSE and log-likelihood variance; noise sensitivity; empirical N-particles vs τ grid study (RMSE and log-likelihood heatmaps, KF floor comparison); resampling method comparison; `LinearTSSM` misspecification test; `LinearARMASSM` |
| `parameter_estimation.ipynb` | MLE via Kalman log-likelihood; MH and PMMH posterior inference; effect of N_particles on PMMH (α·σ identification ridge); effect of observation noise on parameter recoverability; model misspecification (Gaussian estimator on t or ARMA data) |
| `hmc_estimation.ipynb` | HMC vs MH vs MLE on `SimpleLinearGaussianSSM`: comparison table, trace plots, posterior density overlays, autocorrelation and effective sample size (ESS), joint scatter plots |
| `basic_results.ipynb` | Empirical study of `SimpleLinearGaussianSSM`: filtering (PF vs KF), MLE, MH with free and fixed α, weak-identifiability ridge, N=100 multi-trial RMSE and hypothesis tests |
| `macro_data_linear.ipynb` | LG-SSMs on US macro FRED data: simple LG-SSM, multivariate factor model (GDP, IP, ΔUR), HLW macro model (output gap, trend growth, NAIRU, neutral rate), free LG-SSM; MLE estimation and Kalman filter/RTS smoother for each |

---

## Legacy code — `src/older_code/`

Earlier monolithic implementations, kept for reference. Superseded by the `models/` + `estimation/` architecture.

| File | Description |
|------|-------------|
| `particle_filter.py` | Standalone bootstrap PF functions |
| `state_space_model.py` | Combined SSM + PMMH in a single file |
| `regime_switching.py` | Two-regime bootstrap PF + PMMH |
| `rbpf.py` | Rao-Blackwellized PF for a 9-parameter growth model |
| `kim_filter.py` | Kim filter + MLE for the growth model |
| `regime_change_macro.py` | Full macro model with Kim filter, smoother, and forecast |
