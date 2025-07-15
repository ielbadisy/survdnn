
# survdnn <img src="https://raw.githubusercontent.com/ielbadisy/survdnn/main/inst/logo.png" align="right" height="140"/>

> Deep Neural Networks for Survival Analysis Using
> [torch](https://torch.mlverse.org/)

[![License:
MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![R-CMD-check](https://github.com/ielbadisy/survdnn/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/ielbadisy/survdnn/actions/workflows/R-CMD-check.yaml)

------------------------------------------------------------------------

`survdnn` implements neural network-based models for right-censored
survival analysis using the native `torch` backend in R. It supports
multiple loss functions including Cox partial likelihood, L2-penalized
Cox, Accelerated Failure Time (AFT) objectives, as well as
time-dependent extension such as Cox-Time. The package provides a
formula interface, supports model evaluation using time-dependent
metrics (e.g., C-index, Brier score, IBS), cross-validation, and
hyperparameter tuning.

------------------------------------------------------------------------

## Features

- Formula interface for `Surv() ~ .` models
- Modular neural architectures: configurable layers, activations, and
  losses
- Built-in survival loss functions:
  - `"cox"`: Cox partial likelihood
  - `"aft"`: Accelerated Failure Time
  - `"cox_l2"`: penalized Cox
  - `"coxtime"`: deep time-dependent Cox (like DeepSurv)
- Evaluation: C-index, Brier score, Integrated Brier Score (IBS)
- Model selection with `cv_survdnn()` and `tune_survdnn()`
- Prediction of survival curves via `predict()` and `plot()`
- GPU compatible (coming soon)

------------------------------------------------------------------------

## Installation

``` r
# Install from GitHub
# install.packages("remotes")
#remotes::install_github("ielbadisy/survdnn")

# Or clone and install locally
# git clone https://github.com/ielbadisy/survdnn.git
# setwd("survdnn")
# devtools::install()
```

------------------------------------------------------------------------

## Quick Example

``` r
library(survdnn)
library(survival)
```

    ## 
    ## Attaching package: 'survival'

    ## The following object is masked from 'package:survdnn':
    ## 
    ##     brier

``` r
library(ggplot2)

data(veteran, package = "survival")
```

    ## Warning in data(veteran, package = "survival"): data set 'veteran' not found

``` r
mod <- survdnn(
  Surv(time, status) ~ age + karno + celltype,
  data = veteran,
  hidden = c(32, 16),
  epochs = 100,
  loss = "cox",
  verbose = TRUE
)
```

    ## Epoch 50 - Loss: 3.985620
    ## Epoch 100 - Loss: 3.923990

``` r
summary(mod)
```

    ## 

    ## ── Summary of survdnn model ────────────────────────────────────────────────────────────────────

    ## 
    ## Formula:
    ##   Surv(time, status) ~ age + karno + celltype
    ## <environment: 0x55742a26de10>
    ## 
    ## Model architecture:
    ##   Hidden layers:  32 : 16 
    ##   Activation:  relu 
    ##   Dropout:  0.3 
    ##   Final loss:  3.923990 
    ## 
    ## Training summary:
    ##   Epochs:  100 
    ##   Learning rate:  1e-04 
    ##   Loss function:  cox 
    ## 
    ## Data summary:
    ##   Observations:  137 
    ##   Predictors:  age, karno, celltypesmallcell, celltypeadeno, celltypelarge 
    ##   Time range: [ 1, 999 ]
    ##   Event rate:  93.4%

``` r
plot(mod, group_by = "celltype", times = 1:300)
```

![](README_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

------------------------------------------------------------------------

## Loss Functions

``` r
# Cox partial likelihood
mod1 <- survdnn(
  Surv(time, status) ~ age + karno,
  data = veteran,
  loss = "cox",
  epochs = 100
)
```

    ## Epoch 50 - Loss: 3.912308
    ## Epoch 100 - Loss: 3.932914

``` r
# Accelerated Failure Time
mod2 <- survdnn(
  Surv(time, status) ~ age + karno,
  data = veteran,
  loss = "aft",
  epochs = 100
)
```

    ## Epoch 50 - Loss: 12.996538
    ## Epoch 100 - Loss: 12.497688

``` r
# Deep time-dependent Cox (Coxtime)
mod3 <- survdnn(
  Surv(time, status) ~ age + karno,
  data = veteran,
  loss = "coxtime",
  epochs = 100
)
```

    ## Epoch 50 - Loss: 4.905575
    ## Epoch 100 - Loss: 4.867847

------------------------------------------------------------------------

## Cross-Validation

``` r
cv_results <- cv_survdnn(
  Surv(time, status) ~ age + karno + celltype,
  data = veteran,
  times = c(30, 90, 180),
  metrics = c("cindex", "ibs"),
  folds = 3,
  hidden = c(16, 8),
  loss = "cox",
  epochs = 100
)
```

    ## Epoch 50 - Loss: 3.653896
    ## Epoch 100 - Loss: 3.561917
    ## Epoch 50 - Loss: 3.470510
    ## Epoch 100 - Loss: 3.488897
    ## Epoch 50 - Loss: 3.684512
    ## Epoch 100 - Loss: 3.586723

``` r
print(cv_results)
```

    ## # A tibble: 6 × 3
    ##    fold metric value
    ##   <int> <chr>  <dbl>
    ## 1     1 cindex 0.482
    ## 2     1 ibs    0.245
    ## 3     2 cindex 0.714
    ## 4     2 ibs    0.230
    ## 5     3 cindex 0.618
    ## 6     3 ibs    0.213

------------------------------------------------------------------------

## Hyperparameter Tuning

``` r
grid <- list(
  hidden     = list(c(16), c(32, 16)),
  lr         = c(1e-3),
  activation = c("relu"),
  epochs     = c(100),
  loss       = c("cox", "aft")
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
```

    ## Epoch 50 - Loss: 14.174407
    ## Epoch 100 - Loss: 9.958608
    ## Epoch 50 - Loss: 15.507205
    ## Epoch 100 - Loss: 12.294044
    ## Epoch 50 - Loss: 17.522932
    ## Epoch 100 - Loss: 13.783732
    ## Epoch 50 - Loss: 3.426694
    ## Epoch 100 - Loss: 3.342683
    ## Epoch 50 - Loss: 3.485209
    ## Epoch 100 - Loss: 3.469287
    ## Epoch 50 - Loss: 3.434398
    ## Epoch 100 - Loss: 3.350627
    ## Epoch 50 - Loss: 15.841441
    ## Epoch 100 - Loss: 12.313823
    ## Epoch 50 - Loss: 14.313324
    ## Epoch 100 - Loss: 10.436322
    ## Epoch 50 - Loss: 12.376007
    ## Epoch 100 - Loss: 9.313435
    ## Epoch 50 - Loss: 3.329328
    ## Epoch 100 - Loss: 3.235596
    ## Epoch 50 - Loss: 3.398006
    ## Epoch 100 - Loss: 3.433734
    ## Epoch 50 - Loss: 3.410466
    ## Epoch 100 - Loss: 3.352182

``` r
print(tune_res)
```

    ## # A tibble: 4 × 8
    ##   hidden       lr activation epochs loss  metric  mean     sd
    ##   <list>    <dbl> <chr>       <dbl> <chr> <chr>  <dbl>  <dbl>
    ## 1 <dbl [2]> 0.001 relu          100 cox   cindex 0.736 0.0293
    ## 2 <dbl [1]> 0.001 relu          100 cox   cindex 0.727 0.0665
    ## 3 <dbl [2]> 0.001 relu          100 aft   cindex 0.686 0.0412
    ## 4 <dbl [1]> 0.001 relu          100 aft   cindex 0.563 0.135

------------------------------------------------------------------------

## Plot Survival Curves

``` r
plot(mod1, group_by = "celltype", times = 1:300)
```

![](README_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
plot(mod1, group_by = "celltype", times = 1:300, plot_mean_only = TRUE)
```

![](README_files/figure-gfm/unnamed-chunk-6-2.png)<!-- -->

------------------------------------------------------------------------

## Documentation

``` r
help(package = "survdnn")
?survdnn
?tune_survdnn
?cv_survdnn
?plot.survdnn
```

------------------------------------------------------------------------

## Testing

``` r
# Run all tests
devtools::test()
```

------------------------------------------------------------------------

## Contributions

Contributions, issues, and feature requests are welcome. Open an
[issue](https://github.com/ielbadisy/survdnn/issues) or submit a pull
request!

------------------------------------------------------------------------

## License

MIT © [Imad El Badisy](mailto:elbadisyimad@gmail.com)
