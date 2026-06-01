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
│   │   ├── linear_gaussian.py         # LinearGaussianSSM, SimpleLinearGaussianSSM, FixedAlphaSSM
│   │   ├── linear_model_notes.md      # Kalman filter derivations, score, HMC parameterization
│   │   ├── linear_t.py                # LinearTSSM (t-distributed process noise)
│   │   ├── linear_ARMA.py             # LinearARMASSM (ARMA(1,3) latent state)
│   │   ├── linear_factor.py           # MultivariateObservationLGSSM (single-factor model)
│   │   ├── linear_macro.py            # LinearMacroSSM (HLW-style macro model)
│   │   ├── linear_macro_model.md      # HLW macro model derivations and parameter descriptions
│   │   ├── regime_switching_base.py   # RegimeSwitchingBase, RegimeSwitchingDims, RegimeSwitchingStructure
│   │   └── regime_switching_simple.py # SimpleRegimeSwitchingSSM, FixedAlphaRS
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
│   │   ├── nelder_mead.py             # Two-stage Nelder-Mead PMMLE (derivative-free)
│   │   └── kde.py                     # Weighted KDE for PF posteriors and MCMC chains
│   ├── utils.py                       # Shared numerical utilities
│   └── time_series_analysis.py        # ARIMA grid search and BIC comparison
├── notebooks/
│   ├── extension_results.ipynb        # Main results notebook (nine sections; see below)
│   └── hmc_estimation.ipynb           # HMC vs MH vs MLE comparison
├── data/                              # Data acquisition scripts and cached CSVs
└── results/                           # Output files from multi-trial runs (CSV checkpoints)
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
| `.update_params(constrained_params)` | Update model attributes in-place; called by MCMC each iteration |
| `.clear_state()` | Reset accumulated mutable runtime state; default no-op |
| `.constrain_params(theta_unc)` | Map unconstrained vector → valid parameter object |
| `.unconstrain_params(constrained_params)` | Inverse; returns flat `np.ndarray` |
| `.describe()` | Human-readable model summary with equations |

Note: MCMC samplers in unconstrained parameter space may suffer from slower mixing when true parameters are close to a boundary.

---

### `linear_gaussian.py`

| Class | Model |
|-------|-------|
| `LinearGaussianSSM(a, c, q, r, b, d, mu_0, p_0)` | `x_t = A x_{t-1} + b + ε_t`, `y_t = C x_t + d + ν_t`; general multivariate; initial distribution defaults to stationary via `solve_discrete_lyapunov` |
| `SimpleLinearGaussianSSM(phi, alpha, sigma2, tau2)` | `x_t = φ x_{t-1} + ε_t`, `y_t = α x_t + ν_t`; 1-D latent, 1-D observation, Gaussian noise; parameters are **variances** `σ²`, `τ²` |
| `FixedAlphaSSM(alpha_fixed, phi, sigma2, tau2)` | `SimpleLinearGaussianSSM` with `α` fixed at a known value; free parameters are `(φ, σ², τ²)`; removes the α–σ² scale ambiguity to restore identifiability |

Both `SimpleLinearGaussianSSM` and `LinearGaussianSSM` implement the full `StateSpaceModel` interface including `update_params`, `constrain_params` / `unconstrain_params`, and `log_likelihood(data)` (Kalman filter recursion, exact marginal likelihood).

`SimpleLinearGaussianSSM` additionally implements:

| Method | Description |
|--------|-------------|
| `.score(data)` | Analytic gradient `∇_θ log p(y\|θ)` w.r.t. `(φ, α, σ², τ²)`; propagates sensitivities through the Kalman recursion in a single O(T) forward pass |
| `.jacobian_constrain_params(u)` | Diagonal Jacobian `dθ/du` of the constrain transform; used by HMC for the chain rule `∇_u ℓ = diag(J) ⊙ ∇_θ ℓ` |

`FixedAlphaSSM` overrides `score` to strip the fixed-α component, returning gradients only w.r.t. the three free parameters.

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

### `regime_switching_base.py`

Base class for Markov-switching linear Gaussian SSMs. Exposes the interface required by `KimFilter` and `RaoBlackwellizedParticleFilter`.

| Symbol | Description |
|--------|-------------|
| `RegimeSwitchingDims` | Dataclass: `n_regimes`, `state_dim`, `obs_dim` |
| `RegimeSwitchingStructure` | Dataclass: flags controlling which matrices vary by regime (`regime_specific_A/C/Q/R`, `has_state_intercept`, `has_obs_intercept`) |
| `RegimeSwitchingBase` | Base class; stores `A_list, C_list, Q_list, R_list, b_list, d_list`, `regime_transition_matrix`, `regime_probabilities_stationary`; implements `log_likelihood_given_regimes(y, regime_path)` via per-particle Kalman filter |

---

### `regime_switching_simple.py`

Simple 1-D Markov-switching linear Gaussian SSM where only process noise is regime-dependent; `φ`, `α`, and `τ²` are shared across regimes.

```
Transition:  x_t = φ x_{t-1} + ε_t,   ε_t ~ N(0, σ²_{s_t})
Observation: y_t = α x_t + ν_t,        ν_t ~ N(0, τ²)
Regime:      s_t | s_{t-1} ~ Categorical(P[s_{t-1}, :])
```

| Class | Description |
|-------|-------------|
| `SimpleRegimeSwitchingSSM(phi, alpha, sigma2, tau2, trans_matrix)` | K-regime model; `sigma2` is a length-K array of per-regime variances; inherits from `RegimeSwitchingBase` so the RBPF and KimFilter accept it directly |
| `FixedAlphaRS(alpha_fixed, phi, sigma2, tau2, trans_matrix)` | `SimpleRegimeSwitchingSSM` with `α` fixed; free parameters are `(φ, σ²_0, …, σ²_{K-1}, τ², P)` |

Flat parameter layout (length `3 + K + K²`): constrained `[φ, α, σ²_0, …, σ²_{K-1}, τ², P_{00}, …, P_{K-1,K-1}]`; unconstrained uses arctanh for `φ`, log for variances, and row-wise softmax logits for `P`.

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

### `nelder_mead.py`

Two-stage derivative-free Particle Marginal Maximum Likelihood Estimator (PMMLE). Uses Nelder-Mead to maximize the particle-filter log-likelihood, which is too noisy for gradient-based methods.

| Symbol | Description |
|--------|-------------|
| `NelderMeadPMMLE(model, data, N_particles_1, N_particles_2, n_restarts, resample_method, seed)` | Two-stage optimizer: stage 1 runs `n_restarts` coarse searches at `N_particles_1`; stage 2 refines from the best result at `N_particles_2` |
| `.fit(theta0)` | Returns `PMMResult` |
| `PMMResult` | Dataclass: `constrained_params`, `unconstrained_params`, `loglik`, `success_1`, `success_2`, `n_evals_1`, `n_evals_2`, `message`; `.summary()` prints a parameter table |

Within each stage the PF seed is fixed, making the objective deterministic in θ for that run (common random numbers trick). A fresh seed in stage 2 prevents overfitting to the stage-1 noise realization.

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

### `extension_results.ipynb` — Main results notebook

Nine-section empirical study covering filtering, estimation, and model extensions.

| Section | Contents |
|---------|----------|
| I: Basic Results (Filtering) | KF vs PF on synthetic `SimpleLinearGaussianSSM` data; RMSE and log-likelihood comparison |
| II: PF Detailed Results | Effect of N on RMSE and log-likelihood variance (std ∝ 1/√N); noise sensitivity (σ²/τ² sweep); resampling method comparison (systematic vs stratified vs residual vs multinomial) |
| III: Basic Results (Estimation) | α–σ² scale-ambiguity ridge visualization; MLE (L-BFGS-B, delta-method std errors); MH posterior; Nelder-Mead PMMLE (two-stage); HMC; four-method comparison table |
| IV: MCMC Diagnostics | Posterior mean convergence vs chain length; effect of misspecified fixed α; credible intervals and likelihood ratio test |
| V: PMMH and Blocked PMMH | PMMH vs BlockPMMH comparison; autocorrelation and ESS |
| VI: Regime-Switching Filtering | `SimpleRegimeSwitchingSSM` synthetic data; RBPF regime detection vs bootstrap PF; particle cloud visualization |
| VII: Regime-Switching Estimation | Nelder-Mead RBPF-PMLE; RBPF-PMMH posterior inference over `(φ, σ²_0, σ²_1, τ², P)` |
| IX: Model Misspecification | LG estimator applied to Student-t and ARMA data; parameter bias and log-likelihood penalty quantification |

### `hmc_estimation.ipynb`

HMC vs MH vs MLE on `SimpleLinearGaussianSSM`: comparison table, trace plots, posterior density overlays, autocorrelation and effective sample size (ESS), joint scatter plots.
