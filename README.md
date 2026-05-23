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
│   │   ├── base.py                   # Abstract StateSpaceModel base class
│   │   ├── linear_gaussian.py        # SimpleLinearGaussianSSM, LinearGaussianSSM
│   │   ├── linear_t.py               # LinearTSSM (t-distributed process noise)
│   │   ├── linear_ARMA.py            # LinearARMASSM (ARMA(1,3) latent state)
│   │   ├── regime_switching.py       # RegimeSwitchingSSM (general K-regime)
│   │   └── regime_switching_macro.py # RegimeSwitchingMacro (6-state macro model)
│   ├── estimation/
│   │   ├── resampling_methods.py     # Resampling schemes for the particle filter
│   │   ├── particle_filter.py        # Bootstrap particle filter
│   │   ├── kalman_filter.py          # Kalman filter + RTS smoother
│   │   ├── pmmh.py                   # PMMH and BlockPMMH
│   │   ├── kim_filter.py             # Kim filter (in progress)
│   │   └── mle_estimator.py          # MLE via Kalman likelihood (in progress)
│   ├── utils.py                      # Shared numerical utilities
│   └── older_code/                   # Legacy implementations (superseded)
├── notebooks/
│   └── testing_estimation.ipynb      # Filter comparisons, N-particle sweep, resampling comparison
└── data/
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

---

### `linear_gaussian.py`

| Class | Model |
|-------|-------|
| `SimpleLinearGaussianSSM(phi, alpha, sigma, tau)` | `x_t = φ x_{t-1} + ε_t`, `y_t = α x_t + ν_t`; 1-D latent, 1-D observation, Gaussian noise |
| `LinearGaussianSSM(a, c, q, r, b, d, mu_0, p_0)` | `x_t = A x_{t-1} + b + ε_t`, `y_t = C x_t + d + ν_t`; general multivariate; initial distribution defaults to stationary via `solve_discrete_lyapunov` |

Both implement the full `StateSpaceModel` interface including `update_params`.

---

### `linear_t.py`

| Class | Model |
|-------|-------|
| `LinearTSSM(alpha, tau, phi, sigma, df)` | Same observation equation as `SimpleLinearGaussianSSM` but process noise is Student-t; useful for testing filter robustness to heavy tails |

---

### `linear_ARMA.py`

| Class | Model |
|-------|-------|
| `LinearARMASSM(phi, alpha, c, theta_1, theta_2, theta_3, sigma, tau)` | ARMA(1,3) latent process lifted to Markov state `s_t = [x_t, ν_t, ν_{t-1}, ν_{t-2}]`; 4-D latent, 1-D observation |

Overrides `clear_state()` to reset `s` and `s_history` between particle filter runs.

---

### `regime_switching.py`

| Class | Model |
|-------|-------|
| `RegimeSwitchingSSM(A_list, C_list, Q_list, R_list, regime_transition_matrix)` | General K-regime Markov-switching linear Gaussian SSM; per-regime matrices `A_k, C_k, Q_k, R_k`; initial regime drawn from stationary distribution of the Markov chain |

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

### `pmmh.py`

Particle Marginal Metropolis-Hastings. Requires the model to implement `constrain_params`, `unconstrain_params`, and `update_params`.

| Symbol | Description |
|--------|-------------|
| `PMMH(model, particle_filter, n_iter, step_sizes, theta0, log_prior, seed)` | Standard PMMH; Gaussian random walk in unconstrained space |
| `.run()` | Returns `(chain, loglik_chain, accepted)`; `chain` has shape `(n_iter+1, d)` |
| `BlockPMMH(..., blocks)` | Block-update PMMH; cycles through parameter index groups each iteration; each block is accepted/rejected independently |

`theta0` must be in unconstrained space. Use `model.unconstrain_params(...)` to obtain the initial vector. The particle filter's history is cleared automatically before each likelihood evaluation.

---

### `kim_filter.py` *(in progress)*

Kim (1994) approximate filter for regime-switching models with linear Gaussian structure. Reduces Monte Carlo variance by marginalizing the continuous state analytically.

---

### `mle_estimator.py` *(in progress)*

Maximum likelihood estimation via the Kalman-filter log-likelihood for linear Gaussian models.

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

## Notebooks

| Notebook | Contents |
|----------|----------|
| `testing_estimation.ipynb` | Single-run particle filter; Monte Carlo RMSE; effect of particle count on RMSE and log-likelihood variance; noise sensitivity; resampling method comparison; `LinearTSSM` misspecification test; `LinearARMASSM` |

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
