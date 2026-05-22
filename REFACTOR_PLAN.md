# Refactor Plan

## Identified duplications

### 1. ~~`logsumexp`, `log_normal_pdf_scalar`, `systematic_resample`~~ ✓ Done

All three are now defined exactly once in `src/utils.py`:
- `logsumexp` — scalar return by default, optional `axis` parameter
- `log_normal_pdf_scalar` — includes `var ≤ 0` guard
- `log_normal_pdf` — vectorized sd-based variant (also moved here)
- `systematic_resample(weights, rng)` — explicit-RNG version only; the buggy `_systematic_resample` (used global numpy state) was deleted

All former definition sites (`state_space_model.py`, `regime_switching.py`, `particle_filter.py`) now import from `utils`. Import lines in `rbpf.py` and `kim_filter.py` were updated accordingly.

---

### 2. `_log_normal_kernel` — defined in 2 places

| File | Line |
|------|------|
| `src/regime_switching.py` | 309–311 |
| `src/rbpf.py` | 203–205 |

Both are identical: `-0.5 * ((x - mean) / sd) ** 2`.

**Fix:** Move to `utils.py`, import in both files.

---

### 3. `sigmoid` / `logit` — belong in `utils.py`

Currently defined in `regime_switching.py`, imported by `rbpf.py`.

**Fix:** Move to `utils.py`. Update imports in `regime_switching.py` and `rbpf.py`.

---

### 4. PMMH loop — nearly identical in 3 files

The core Metropolis-Hastings accept/reject loop appears in:

- `src/state_space_model.py` — `pmmh()`
- `src/regime_switching.py` — `pmmh_regime_switching()`
- `src/rbpf.py` — `pmmh_rbpf()`

All three follow the same pattern:
```python
z_prop = z + rng.normal(0, step_sizes)
lp_prior_prop = log_prior(z_prop)
if isfinite(lp_prior_prop):
    log_lik_prop = pf_log_lik(y, constrain(z_prop), N, seed)
    log_post_prop = log_lik_prop + lp_prior_prop
    if log(rng.uniform()) < log_post_prop - log_post:
        z, log_post, log_lik = z_prop, log_post_prop, log_lik_prop
        accepts += 1
```

**Fix:** Absorbed into the generic `estimation/pmmh.py` module described below.

---

### 5. Kalman predict/update — two incompatible implementations

| File | Interface |
|------|-----------|
| `src/kim_filter.py` | `kalman_predict_update(m, C, y_t, regime_j, theta)` — combined; builds matrices internally |
| `src/regime_change_macro.py` | `kalman_predict(m, P, A, a, Q)` + `kalman_update(m_pred, P_pred, y_t, H, b, R)` — split; accepts explicit matrices |

**Fix:** Promote the split-step versions to `src/kalman.py`. Rewrite `kim_filter.py:kalman_predict_update` to call them after building matrices via `build_matrices`.

---

### 6. Gaussian mixture collapsing — inline vs. function

`regime_change_macro.py` has `collapse_gaussian_mixture(means, covs, weights)`. The same weighted-mean + covariance computation is done inline in `rbpf.py` and `kim_filter.py`.

**Fix:** Move `collapse_gaussian_mixture` to `utils.py`. Replace inline loops with imports.

---

### 7. `StateSpaceModel` stub class in `particle_filter.py`

`particle_filter.py` declares an empty `StateSpaceModel` that shadows the real abstract class from `state_space_model.py`.

**Fix:** Delete the stub. Absorbed into the `models/` layer described below.

---

## Proposed module structure after refactor

The target architecture separates **model definitions** (what a model is) from **estimation algorithms** (how to fit it). Estimation functions receive a model instance and data — they never hard-code model-specific matrices or priors.

```
src/
├── utils.py                        # [DONE] Shared math primitives
│   ├── logsumexp(a, axis)
│   ├── log_normal_pdf_scalar(y, mean, var)
│   ├── log_normal_pdf(y, mean, sd)
│   ├── systematic_resample(weights, rng)
│   ├── softmax(x)
│   ├── sigmoid(z) / logit(p)       # move from regime_switching.py
│   ├── _log_normal_kernel(x, m, s) # move from regime_switching.py / rbpf.py
│   └── collapse_gaussian_mixture(means, covs, weights)
│
├── kalman.py                       # Kalman building blocks
│   ├── kalman_predict(m, P, A, a, Q) -> (m_pred, P_pred)
│   └── kalman_update(m_pred, P_pred, y, H, b, R) -> (m_filt, P_filt, loglik, ...)
│
├── models/
│   │
│   ├── base.py                     # Abstract StateSpaceModel interface
│   │   └── class StateSpaceModel
│   │       ├── log_prior(z) -> float               # prior over unconstrained params
│   │       ├── constrain(z) -> theta               # unconstrained → constrained
│   │       ├── unconstrain(theta) -> z             # constrained → unconstrained
│   │       ├── transition(x_prev, regime, rng)     # sample x_t | x_{t-1}
│   │       ├── observation(x, regime, rng)         # sample y_t | x_t
│   │       ├── log_transition_density(x_next, x_prev, regime) -> float
│   │       └── log_observation_density(y, x, regime) -> float
│   │
│   ├── linear_gaussian.py          # AR(1) model, 4 params (phi, alpha, sigma, tau)
│   │   └── class LinearGaussianSSM(StateSpaceModel)
│   │
│   ├── regime_switching.py         # 2-regime volatility model, 6 params
│   │   └── class TwoRegimeSSM(StateSpaceModel)
│   │       # holds: stationary_regime_probs, build_matrices, default_initial_state
│   │       # holds: constrain/unconstrain, log_prior (moved from regime_switching.py)
│   │
│   ├── growth_model.py             # 9-param growth model supporting RBPF
│   │   └── class RegimeGrowthModel(StateSpaceModel)
│   │       # holds: constrain/unconstrain, log_prior (moved from rbpf.py)
│   │       # exposes: kalman_step(m, C, y, regime) for RBPF integration
│   │
│   └── macro.py                    # Full 4-observable macro model
│       ├── class RegimeMacroModel(StateSpaceModel)
│       │   └── (moved from regime_change_macro.py)
│       └── dataclasses: ModelDims, RegimeStructure, MacroParams
│
└── estimation/
    │
    ├── particle_filter.py          # Generic bootstrap PF
    │   └── bootstrap_pf(model, y, N_particles, seed, ...) -> dict
    │       # replaces: particle_filter(), bootstrap_pf_regime_switching()
    │       # model supplies log_observation_density + transition
    │
    ├── rbpf.py                     # Rao-Blackwellized PF
    │   └── rao_blackwell_pf(model, y, N_particles, seed, ...) -> dict
    │       # replaces: rbpf_regime_growth()
    │       # model supplies kalman_step() for continuous state integration
    │
    ├── kim_filter.py               # Kim approximate filter and smoother
    │   ├── kim_filter(model, y, covariates) -> KimFilterResult
    │   └── kim_smoother(model, filter_result) -> KimSmootherResult
    │       # replaces: kim_filter_regime_growth(), kim_filter() in regime_change_macro.py
    │       # model supplies A(), Q(), H(), b(), R() matrix methods
    │
    ├── pmmh.py                     # Generic PMMH
    │   └── run_pmmh(model, y, n_iter, N_particles, step_sizes, seed, ...) -> dict
    │       # replaces: pmmh(), pmmh_regime_switching(), pmmh_rbpf()
    │       # model supplies log_prior, constrain, and a log_likelihood(y, theta) method
    │       # internally calls whichever PF the model declares as its likelihood estimator
    │
    └── mle.py                      # Gradient-based MLE
        └── fit_mle(model, y, theta0, n_restarts, compute_se, ...) -> dict
            # replaces: kim_mle(), fit_mle() in regime_change_macro.py
            # model supplies log_prior (for label-switching constraints) + log_likelihood
```

### Key design rules

1. **Models own their parameters.** Each model class holds its own `constrain`/`unconstrain` transforms and `log_prior`. Estimation algorithms never hard-code parameter layouts.

2. **Estimators are generic.** `run_pmmh`, `bootstrap_pf`, `rao_blackwell_pf`, and `fit_mle` accept any model that satisfies the `StateSpaceModel` interface. Swapping a `LinearGaussianSSM` for a `TwoRegimeSSM` requires only changing the model argument.

3. **Kalman steps are reusable.** `kalman.py` provides the low-level `kalman_predict` / `kalman_update` operations. Both `estimation/kim_filter.py` and `models/growth_model.py` (for the RBPF's per-particle Kalman step) import from there.

4. **`utils.py` stays pure.** Only stateless mathematical functions — no model logic, no data structures.

---

## Priority order

| Priority | Change | Status |
|----------|--------|--------|
| 1 | Consolidate `logsumexp`, `log_normal_pdf_scalar`, `systematic_resample` into `utils.py` | ✓ Done |
| 2 | Move `sigmoid`, `logit`, `_log_normal_kernel`, `collapse_gaussian_mixture` into `utils.py` | Pending |
| 3 | Extract `kalman_predict` / `kalman_update` into `kalman.py` | Pending |
| 4 | Create `models/` layer — move class definitions out of `state_space_model.py`, `regime_switching.py`, `rbpf.py`, `regime_change_macro.py` | Pending |
| 5 | Create `estimation/` layer — make PF, RBPF, Kim filter, PMMH, MLE generic over model instances | Pending |
| 6 | Delete stub `StateSpaceModel` / `ResamplingMethod` classes from `particle_filter.py` | Pending |
