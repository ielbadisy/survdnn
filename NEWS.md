# survdnn

## survdnn 0.7.5

### Main changes

* **Fixed and stabilized loss implementations** for AFT and CoxTime models, ensuring correct handling of time scaling, parameter learning, and numerical stability.
* **Corrected and harmonized prediction methods** (`predict.survdnn`) across all supported losses (Cox, Cox L2, AFT, CoxTime), including survival and risk predictions.
* **Improved internal consistency checks** to ensure valid survival probabilities (monotonicity and bounds).
* **Updated and expanded unit tests** to cover prediction behavior, edge cases, and reproducibility.
* **Regenerated Rd documentation** to fully document all function arguments and remove previous documentation warnings.



## survdnn 0.7.0

### Major changes

* Added full support for **training control mechanisms**, including early stopping callbacks and complete loss tracking across epochs.

* Introduced `plot_loss()` to visualize training loss trajectories and diagnose convergence or instability.

* Centralized **reproducibility control** via the `.seed` argument in `survdnn()`, synchronizing both R and Torch random number generators.

* Expanded optimizer support to include **Adam, AdamW, SGD, RMSprop, and Adagrad**, with customizable optimizer arguments.

* Enhanced **prediction methods** to robustly support linear predictors, survival probabilities, and cumulative risk across all supported loss functions.

* Added explicit and user-controllable **missing-data handling** (`na_action = "omit"` or `"fail"`), with informative messages.

### Minor changes

* Improved handling of formulas using `Surv(...) ~ .` in prediction and evaluation.

* Improved printing and summary methods for fitted `survdnn` objects.

* Expanded unit test coverage, including optimizers, plotting utilities, and missing-data edge cases.

### Bug fixes

* Fixed inconsistencies in prediction and evaluation when formulas used `.` expansion.

---

## survdnn 0.6.2

### Maintenance release (CRAN compliance)

- **Removed automatic `torch::install_torch()` on load:**  

  The package no longer downloads or installs Torch libraries automatically when loaded. The `.onLoad()` function now performs only a silent availability check, and `.onAttach()` displays an informative message instructing users to manually run `torch::install_torch()` when necessary.

  This update ensures full compliance with CRAN policies that forbid modification of user environments or network activity during package load.

- Updated startup messages for clearer user guidance.

- Internal documentation updates and version bump for CRAN resubmission.

---

## survdnn 0.6.1

### Infrastructure and testing improvements

- Added conditional test skipping: tests and examples now use `skip_if_not(torch_is_installed())` and `skip_on_cran()` to avoid failures on systems where Torch is not available (thanks to @dfalbel for the [PR](https://github.com/ielbadisy/survdnn/pull/5)).

- Regenerated documentation (`RoxygenNote: 7.3.3`) and updated man pages.

- Minor internal consistency fixes and CI check updates.

---

## survdnn 0.6.0

First public release of `survdnn`.

### Features

- `survdnn()`: Fit deep learning survival models using a formula interface.
- Supported loss functions:
  - Cox partial likelihood (`"cox"`)
  - L2-penalized Cox (`"cox_l2"`)
  - Time-dependent Cox (`"coxtime"`)
  - Accelerated Failure Time (`"aft"`)
- Cross-validation via `cv_survdnn()`.
- Hyperparameter tuning with `tune_survdnn()`.
- Survival probability prediction and curve plotting.
- Evaluation metrics: Concordance index (C-index), Brier score, and Integrated Brier Score (IBS).

CRAN submission prepared, including README, documentation, and automated tests.
