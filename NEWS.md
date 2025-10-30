# survdnn 0.6.3

## survdnn 0.6.2

### Maintenance release (CRAN compliance)

- **Removed automatic `torch::install_torch()` on load:**  
  The package no longer downloads or installs Torch libraries automatically when loaded.  
  The `.onLoad()` function now performs only a silent availability check, and `.onAttach()`  
  displays an informative message instructing users to manually run  
  `torch::install_torch()` when necessary.  
  This update ensures full compliance with CRAN policies that forbid modification of user environments  
  or network activity during package load.

- Updated startup message for clearer user guidance.
- Internal documentation and version bump for CRAN resubmission.

---

## survdnn 0.6.1

### Infrastructure and testing improvements

- Added conditional test skipping: tests and examples now use  
  `skip_if_not(torch_is_installed())` and `skip_on_cran()` to avoid failures  
  on systems where Torch is not available.  
  (Thanks to @dfalbel for the PR.)

- Regenerated documentation (`RoxygenNote: 7.3.3`) and updated man pages.
- Minor internal consistency fixes and CI checks update.

---

## survdnn 0.6.0

First public release of `survdnn` — Deep Neural Networks for Survival Analysis in R using `torch`.

### Features

- `survdnn()`: Fit deep learning survival models using a formula interface.
- Supported loss functions:  
  - Cox partial likelihood (`"cox"`)  
  - L2-penalized Cox (`"cox_l2"`)  
  - Time-dependent Cox (`"coxtime"`)  
  - Accelerated Failure Time (`"aft"`)
- Cross-validation with `cv_survdnn()`.
- Hyperparameter tuning with `tune_survdnn()`.
- Survival probability prediction and curve plotting.
- Evaluation metrics: Concordance index (C-index), Brier score, and Integrated Brier Score (IBS).

CRAN submission prepared. Includes README, documentation, and automated tests.
