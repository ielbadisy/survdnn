# survdnn 0.6

## First release

- Implements `survdnn()` for deep neural network-based survival analysis using the `torch` backend in R.
- Supports multiple loss functions: Cox partial likelihood (`"cox"`), L2-penalized Cox (`"cox_l2"`), Accelerated Failure Time (`"aft"`), and rank-based `"coxtime"`.
- Unified prediction interface via `predict.survdnn()` returning time-dependent survival probabilities.
- Includes utilities for:
  - Model evaluation (C-index, Brier score, IBS)
  - Cross-validation: `cv_survdnn()`
  - Hyperparameter tuning: `tune_survdnn()`
- Clean formula interface with support for tabular data input.
- Fully documented with examples and tested unit coverage.

This is the initial public release.
