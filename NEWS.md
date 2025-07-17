## survdnn 0.6.0

First public release of `survdnn` (Deep Neural Networks for Survival Analysis in R using `torch`).

### Features

- `survdnn()`: Fit deep learning survival models using a formula interface
- Supported loss functions: Cox partial likelihood (`"cox"`), L2-penalized Cox (`"cox_l2"`); Time-dependent Cox (`"coxtime"`), and Accelerated Failure Time (`"aft"`)
- Cross-validation with `cv_survdnn()`
- Hyperparameter tuning with `tune_survdnn()`
- Survival probability prediction and curve plotting
- Evaluation metrics: C-index, Brier score, Integrated Brier Score (IBS)

CRAN submission prepared. README, documentation, and tests included.
