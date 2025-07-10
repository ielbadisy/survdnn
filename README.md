
# survdnn package

## About

Deep Neural Network for Survival Analysis using `torch` in R, with
formula interface, modular loss functions, and S3 support.

# Developement TODO list

## CORE IMPLEMENTATION

### Create `survdnn()` with a clean formula interface

- [x] Implement `survdnn()` with formula, data, and key parameters
  (`hidden`, `lr`, `activation`, `epochs`, `.loss_fn`)

- [x] Support `Surv(time, status)`

- [ ] Add argument checks and clear error messages (missing or invalid
  input)

### Implement default Cox loss

- [x] Provide clean `cox_loss()` implementation (default)

- [ ] Validate gradients using a numerical approximation

### Expose `.loss_fn` argument

- [x] Allow `.loss_fn = cox_loss` as default

- [x] Assert `.loss_fn(pred, y)` returns scalar tensor and correct shape

### Allow user-defined loss function

- [ ] Accept any `.loss_fn(pred, true)` that returns a scalar tensor
- [ ] Add internal check and fallback error
- [ ] Provide usage template in documentation

``` r
survdnn(..., .loss_fn = my_custom_loss)
```

### Ensure torch compatibility

- [ ] Internally use `torch_tensor()` with proper `dtype`

- [ ] Validate input types: numeric/double only

- [ ] Avoid base R where torch ops exist (use `torch_mean`, not `mean`)

## LOSS FUNCTION MODULE

### Implement additional loss functions

- [x] `cox_l2_loss` (`cox_loss + λ‖pred‖²`)

- [x] `aft_loss` (MSE of log-times with dist = “weibull”, “lognormal”)

- [x] `rank_loss` (pairwise concordance)

### Document all loss functions

- Add math and use-cases for each:

  - [ ] Cox

  - [ ] Cox-L2

  - [ ] AFT

  - [ ] Rank

- [ ] Export and document with Roxygen2

## PREDICTION AND INTERFACE

### Implement S3 predict method

- [x] Add `predict.survdnn()` using standard S3 interface

- Support:

  - [x] `type = "survival"` (matrix of survival probabilities)

  - [x] `type = "lp"` (linear predictors)

  - [x] `type = "risk"` (`1 - S(t)` at a given time)

- [ ] Document predict interface and return values

### User-friendly printing and summary

- [x] Implement `print.survdnn()` (show architecture, loss, epochs)

- [x] Implement `summary.survdnn()` (show model structure, training
  info, final loss, data summary)

- [x] Implement `plot.survdnn()` for survival curves

## TESTING AND VALIDATION

### Validate model output

- [ ] Train model on benchmark datasets: `veteran`, `lung`, etc.

- [ ] Compare `cox_loss` output to `coxph()` log-likelihood

- [ ] Visual check: survival curves (via `matplot()`)

### Write test suite

- [ ] Use `testthat` to validate:

  - Model fitting

  - Predictions

  - Custom loss integration

  - S3 dispatch

- [ ] Add edge case tests (NA values, invalid loss format)

## MODEL SELECTION AND TUNING

### Implement tuning

- [x] Create `tune_survdnn()` (grid search)

- [ ] Support custom `.loss_fn` in tuning

- [ ] Add support for:

  - Parallel execution

  - Returning best model vs full grid

  - (optional) Plotting

### Implement cross-validation

- [x] Add `cv_survdnn()`:

  - K-fold CV

  - Metric support: C-index, IBS

## DOCUMENTATION + DEMO

### Vignettes and examples

- [ ] Provide a full training example:

  - Fit with Cox loss

  - Fit with custom loss

  - Plot survival predictions

  - Benchmark vs coxph, RSF, DeepHit, pycox, torchsurv, …

## Phase 2 Enhancements

- [ ] GPU/CPU toggle

- [ ] Learning curve plotting

- [ ] Early stopping or validation loss tracking

- [ ] AFT + time-to-event distribution estimation

- [ ] Integration with `survalis` as `fit_survdnn()` and
  `predict_survdnn()`

## Functions to be implemented in the `survdnn` R package

### Core API

| Function            | Purpose                                                   | Exported |
|---------------------|-----------------------------------------------------------|----------|
| `survdnn()`         | Main model-fitting function with formula interface        | Yes      |
| `predict.survdnn()` | Predict method (S3): returns survival, risk, or LP        | Yes      |
| `print.survdnn()`   | Print method (S3): shows architecture and training info   | Yes      |
| `summary.survdnn()` | Summary method (S3): model details, final loss, data info | Yes      |
| `plot.survdnn()`    | Plot survival curves (optional: mean, CI)                 | Yes      |

### Loss functions

| Function             | Purpose                                                   | Exported |
|----------------------|-----------------------------------------------------------|----------|
| `cox_loss()`         | Partial likelihood (default loss)                         | Yes      |
| `cox_l2_loss()`      | Cox loss + L2 penalty on predictions                      | Yes      |
| `aft_loss()`         | AFT loss (e.g., Weibull/log-normal log-likelihood or MSE) | Yes      |
| `rank_loss()`        | Pairwise ranking loss (for concordance)                   | Yes      |
| `flex_loss()`        | Fully parametric log-likelihood of survival distributions | Yes      |
| `validate_loss_fn()` | Internal helper to check that a `.loss_fn` is valid       | No       |

### Training and evaluation

| Function             | Purpose                                                   | Exported |
|----------------------|-----------------------------------------------------------|----------|
| `cv_survdnn()`       | K-fold cross-validation with survival metrics             | Yes      |
| `tune_survdnn()`     | Hyperparameter tuning (grid search, optional parallelism) | Yes      |
| `train_survdnn()`    | Internal training loop (torch-optimized)                  | No       |
| `evaluate_survdnn()` | Internal: evaluation wrapper for model + metric           | Yes      |

### Testing + Benchmarking

| Function                 | Purpose                                          | Exported |
|--------------------------|--------------------------------------------------|----------|
| `benchmark_survdnn()`    | Compare with `coxph()`, `ranger::rfsrc()`, etc.  | Yes      |
| `compare_survdnn_loss()` | Compare `cox_loss()` to `coxph()` log-likelihood | Non      |

### Documentation and examples

| Function/File                | Purpose                                                    | Exported |
|------------------------------|------------------------------------------------------------|----------|
| `vignette("survdnn-intro")`  | Full example: fit, custom loss, predict, plot              | Yes      |
| `vignette("loss-functions")` | Mathematical overview and use-cases for each loss function | Yes      |

### Internal helpers (Not Exported)

| Function               | Purpose                                                   |
|------------------------|-----------------------------------------------------------|
| `.parse_formula()`     | Extract response and predictors from formula              |
| `.torch_dataset()`     | Convert data into torch `dataset()`                       |
| `.get_activation()`    | Map user input to `nn_relu`, `nn_tanh`, etc.              |
| `.compute_surv_prob()` | Compute survival from risk scores (e.g., KM / parametric) |
| `.validate_inputs()`   | Sanity checks on data types and formula                   |
