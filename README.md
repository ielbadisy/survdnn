
# survdnn <img src="https://raw.githubusercontent.com/ielbadisy/survdnn/main/inst/logo.png" align="right" height="140"/>

> Deep Neural Networks for Survival Analysis using [R
> torch](https://torch.mlverse.org/)

[![License:
MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![R-CMD-check](https://github.com/ielbadisy/survdnn/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/ielbadisy/survdnn/actions/workflows/R-CMD-check.yaml)

## About

`survdnn` implements neural network-based models for right-censored
survival analysis using the native `torch` backend in R. It supports
multiple loss functions including Cox partial likelihood, L2-penalized
Cox, Accelerated Failure Time (AFT) objectives, as well as
time-dependent extension such as Cox-Time. The package provides a
formula interface, supports model evaluation using time-dependent
metrics (C-index, Brier score, IBS), cross-validation, and
hyperparameter tuning.

## Review status

A methodological paper describing the design, implementation, and
evaluation of `survdnn` is currently under review at *The R Journal*.

## Main features

- Formula interface for `Surv() ~ .` models

- Modular neural architectures: configurable layers, activations,
  optimizers, and losses

- Built-in survival loss functions:

  - `"cox"`: Cox partial likelihood
  - `"cox_l2"`: penalized Cox
  - `"aft"`: Accelerated Failure Time
  - `"coxtime"`: deep time-dependent Cox

- Evaluation: C-index, Brier score, IBS

- Model selection with `cv_survdnn()` and `tune_survdnn()`

- Prediction of survival curves via `predict()` and `plot()`

## Installation

``` r
# Install from CRAN
install.packages("survdnn")


# Install from GitHub
install.packages("remotes")
remotes::install_github("ielbadisy/survdnn")

# Or clone and install locally
git clone https://github.com/ielbadisy/survdnn.git
setwd("survdnn")
devtools::install()
```

## Quick example

``` r
library(survdnn)
library(survival, quietly = TRUE)
library(ggplot2)

veteran <- survival::veteran

mod <- survdnn(
  Surv(time, status) ~ age + karno + celltype,
  data = veteran,
  hidden = c(32, 16),
  epochs = 300,
  loss = "cox",
  verbose = TRUE
  )

summary(mod)
```

    ## 
    ## Formula:
    ##   Surv(time, status) ~ age + karno + celltype
    ## <environment: 0x6011c835fbe0>
    ## 
    ## Model architecture:
    ##   Hidden layers:  32 : 16 
    ##   Activation:  relu 
    ##   Dropout:  0.3 
    ##   Batch norm:  TRUE 
    ##   Final loss:  3.841008 
    ## 
    ## Training summary:
    ##   Epochs:  300 
    ##   Learning rate:  1e-04 
    ##   Loss function:  cox 
    ##   Optimizer:  adam 
    ##   Device:  cpu 
    ##   CPU threads:  default 
    ##   NA action:  omit 
    ## 
    ## Data summary:
    ##   Observations (used/total): 137 / 137
    ##   Predictors (5): age, karno, celltypesmallcell, celltypeadeno, celltypelarge
    ##   Time range: [ 1, 999 ]
    ##   Events / censored:  128  /  9 
    ##   Event rate:  93.4% 
    ##   Predictors standardized:  yes

``` r
plot(mod, group_by = "celltype", times = 1:300)
```

## Loss Functions

- Cox partial likelihood

``` r
mod1 <- survdnn(
  Surv(time, status) ~ age + karno,
  data = veteran,
  loss = "cox",
  epochs = 300
  )
```

    ## [survdnn::fit] start: n=137 p=2 loss=cox optimizer=adam epochs=300 device=cpu

    ## [survdnn::fit] epoch 50/300 loss=3.989828

    ## [survdnn::fit] epoch 100/300 loss=3.942912

    ## [survdnn::fit] epoch 150/300 loss=3.890888

    ## [survdnn::fit] epoch 200/300 loss=3.869117

    ## [survdnn::fit] epoch 250/300 loss=3.866454

    ## [survdnn::fit] epoch 300/300 loss=3.870147

    ## [survdnn::fit] done: epochs_run=300 final_loss=3.870147

- Accelerated Failure Time

``` r
mod2 <- survdnn(
  Surv(time, status) ~ age + karno,
  data = veteran,
  loss = "aft",
  epochs = 300
  )
```

    ## [survdnn::fit] start: n=137 p=2 loss=aft optimizer=adam epochs=300 device=cpu

    ## [survdnn::fit] epoch 50/300 loss=4.628552

    ## [survdnn::fit] epoch 100/300 loss=4.641005

    ## [survdnn::fit] epoch 150/300 loss=4.563893

    ## [survdnn::fit] epoch 200/300 loss=4.507134

    ## [survdnn::fit] epoch 250/300 loss=4.501204

    ## [survdnn::fit] epoch 300/300 loss=4.494009

    ## [survdnn::fit] done: epochs_run=300 final_loss=4.494009

- Coxtime

``` r
mod3 <- survdnn(
  Surv(time, status) ~ age + karno,
  data = veteran,
  loss = "coxtime",
  epochs = 300
  )
```

    ## [survdnn::fit] start: n=137 p=2 loss=coxtime optimizer=adam epochs=300 device=cpu

    ## [survdnn::fit] epoch 50/300 loss=3.946552

    ## [survdnn::fit] epoch 100/300 loss=3.955686

    ## [survdnn::fit] epoch 150/300 loss=3.944864

    ## [survdnn::fit] epoch 200/300 loss=3.930768

    ## [survdnn::fit] epoch 250/300 loss=3.838923

    ## [survdnn::fit] epoch 300/300 loss=3.889966

    ## [survdnn::fit] done: epochs_run=300 final_loss=3.889966

## Cross-validation

``` r
cv_results <- cv_survdnn(
  Surv(time, status) ~ age + karno + celltype,
  data = veteran,
  times = c(600),
  metrics = c("cindex", "ibs"),
  folds = 3,
  hidden = c(16, 8),
  loss = "cox",
  epochs = 300
  )

print(cv_results)
```

## Hyperparameter tuning

``` r
grid <- list(
  hidden     = list(c(16), c(32, 16)),
  lr         = c(1e-3),
  activation = c("relu"),
  epochs     = c(100, 300),
  loss       = c("cox", "aft", "coxtime")
  )

tune_res <- tune_survdnn(
  formula = Surv(time, status) ~ age + karno + celltype,
  data = veteran,
  times = c(90, 300),
  metrics = "cindex",
  param_grid = grid,
  folds = 3,
  refit = FALSE,
  return = "summary"
  )

print(tune_res)
```

## Tuning and refitting the best Model

`tune_survdnn()` can be used also to automatically refit the
best-performing model on the full dataset. This behavior is controlled
by the `refit` and `return` arguments. For example:

``` r
best_model <- tune_survdnn(
  formula = Surv(time, status) ~ age + karno + celltype,
  data = veteran,
  times = c(90, 300),
  metrics = "cindex",
  param_grid = grid,
  folds = 3,
  refit = TRUE,
  return = "best_model"
  )
```

In this mode, cross-validation is used to select the optimal
hyperparameter configuration, after which the selected model is refitted
on the full dataset. The function then returns a fitted object of class
`"survdnn"`.

The resulting model can be used directly for prediction visualization,
and evaluation:

``` r
summary(best_model)

plot(best_model, times = 1:300)

predict(best_model, veteran, type = "risk", times = 180)
```

This makes `tune_survdnn()` suitable for end-to-end workflows, combining
model selection and final model fitting.

## Plot survival curves

``` r
plot(mod1, group_by = "celltype", times = 1:300)
```

![](README_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

``` r
plot(mod1, group_by = "celltype", times = 1:300, plot_mean_only = TRUE)
```

![](README_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->

## Documentation

``` r
help(package = "survdnn")
?survdnn
?tune_survdnn
?cv_survdnn
?plot.survdnn
```

## Testing

``` r
# run all tests
devtools::test()
```

## Note on reproducibility

By default, `{torch}` initializes model weights and shuffles minibatches
using random draws, so results may differ across runs. Unlike
`set.seed()`, which only controls R’s random number generator, `{torch}`
relies on its own RNG implemented in C++ (and CUDA when using GPUs).

To ensure reproducibility, random seeds must therefore be set at the
Torch level as well.

`survdnn` provides built-in control of randomness to guarantee
reproducible results across runs. The main fitting function,
`survdnn()`, exposes a dedicated `.seed` argument:

``` r
mod <- survdnn(
  Surv(time, status) ~ age + karno + celltype,
  data   = veteran,
  epochs = 300,
  .seed  = 123
)
```

When `.seed` is provided, `survdnn()` internally synchronizes both R and
Torch random number generators via `survdnn_set_seed()`, ensuring
reproducible:

- weight initialization

- dropout behavior

- minibatch ordering

- loss trajectories

If `.seed = NULL` (the default), randomness is left uncontrolled and
results may vary between runs.

For full reproducibility in cross-validation or hyperparameter tuning,
the same `.seed` mechanism is propagated internally by `cv_survdnn()`
and `tune_survdnn()`, ensuring consistent data splits, model
initialization, and optimization paths across repetitions.

## CPU and core usage

`survdnn` relies on the `{torch}` backend for numerical computation. The
number of CPU cores (threads) used during training, prediction, and
evaluation is controlled globally by Torch.

By default, Torch automatically configures its CPU thread pools based on
the available system resources, unless explicitly overridden by the user
using:

``` r
torch::torch_set_num_threads(4)
```

You can also set this directly from `survdnn` APIs with `.threads`:

``` r
mod <- survdnn(
  Surv(time, status) ~ age + karno + celltype,
  data = survival::veteran,
  .threads = 4
)
```

    ## [survdnn::fit] start: n=137 p=5 loss=cox optimizer=adam epochs=300 device=cpu

    ## [survdnn::fit] cpu_threads=4

    ## [survdnn::fit] epoch 50/300 loss=3.962863

    ## [survdnn::fit] epoch 100/300 loss=3.952089

    ## [survdnn::fit] epoch 150/300 loss=3.896662

    ## [survdnn::fit] epoch 200/300 loss=3.834132

    ## [survdnn::fit] epoch 250/300 loss=3.866578

    ## [survdnn::fit] epoch 300/300 loss=3.821199

    ## [survdnn::fit] done: epochs_run=300 final_loss=3.821199

The same `.threads` argument is available in `cv_survdnn()` and
`tune_survdnn()`.

This setting affects:

- model training

- prediction

- evaluation metrics

- cross-validation and hyperparameter tuning

GPU acceleration can be enabled by setting `.device = "cuda"` when
calling `survdnn()` (`cv_survdnn()` and `tune_survdnn()` too).

## Availability

The `survdnn` R package is available on
[CRAN](https://CRAN.R-project.org/package=survdnn) or
[github](https://github.com/ielbadisy/survdnn)

## Contributions

Contributions, issues, and feature requests are welcome!

Open an [issue](https://github.com/ielbadisy/survdnn/issues) or submit a
pull request.

## License

MIT License © 2025 Imad EL BADISY
