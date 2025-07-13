
# survdnn <img src="https://raw.githubusercontent.com/ielbadisy/survdnn/main/inst/logo.png" align="right" height="140"/>

> Deep Neural Networks for Survival Analysis Using
> [torch](https://torch.mlverse.org/)

[![License:
MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![R-CMD-check](https://github.com/ielbadisy/survdnn/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/ielbadisy/survdnn/actions)

`survdnn` implements flexible deep learning models for right-censored
survival data using the powerful `torch` backend in R. It supports Cox,
AFT, and ranking-based objectives, provides a formula interface, and
includes model evaluation, cross-validation, hyperparameter tuning, and
interpretability utilities.

------------------------------------------------------------------------

## Features

- Formula interface for `Surv()` models

- Modular neural architectures with user-defined depth, activation, and
  loss

- Built-in support for:

  - Cox partial likelihood loss
  - Accelerated Failure Time (AFT) loss
  - Ranking loss
  - Custom user-defined losses

- Model evaluation: C-index, Brier score, and IBS.

- Cross-validation and hyperparameter tuning with `cv_survdnn()` and
  `tune_survdnn()`

- Predict survival curves and visualize using `plot()`

- Torch-native, GPU compatible (comming soon)

------------------------------------------------------------------------

## Installation

``` r
# Install from GitHub
remotes::install_github("ielbadisy/survdnn")

# Or clone + install locally
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
data(veteran, package = "survival")
```

    ## Warning in data(veteran, package = "survival"): data set 'veteran' not found

``` r
mod <- survdnn(Surv(time, status) ~ age + karno + celltype,
               data = veteran,
               hidden = c(32, 16),
               epochs = 100,
               verbose = TRUE)
```

    ## Epoch 50 - Loss: 3.905365
    ## Epoch 100 - Loss: 3.947318

``` r
summary(mod)
```

    ## 

    ## ── Summary of survdnn model ────────────────────────────────────────────────────

    ## 
    ## Formula:
    ##   Surv(time, status) ~ age + karno + celltype
    ## <environment: 0x5fa0c4f11608>
    ## 
    ## Model architecture:
    ##   Hidden layers:  32 : 16 
    ##   Activation:  relu 
    ##   Dropout:  0.3 
    ##   Final loss:  3.947318 
    ## 
    ## Training summary:
    ##   Epochs:  100 
    ##   Learning rate:  1e-04 
    ##   Loss function:  cox_loss 
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
# Default: Cox partial likelihood
mod1 <- survdnn(Surv(time, status) ~ age + karno,
                data = veteran,
                .loss_fn = cox_loss,
                epochs = 100)
```

    ## Epoch 50 - Loss: 3.986832
    ## Epoch 100 - Loss: 3.948670

``` r
# AFT loss
mod2 <- survdnn(Surv(time, status) ~ age + karno,
                data = veteran,
                .loss_fn = aft_loss,
                epochs = 100)
```

    ## Epoch 50 - Loss: 15.062006
    ## Epoch 100 - Loss: 14.064785

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
  epochs = 100
  )
```

    ## Epoch 50 - Loss: 3.612589
    ## Epoch 100 - Loss: 3.595535
    ## Epoch 50 - Loss: 3.683477
    ## Epoch 100 - Loss: 3.572265
    ## Epoch 50 - Loss: 3.621128
    ## Epoch 100 - Loss: 3.615796

``` r
print(cv_results)
```

    ## # A tibble: 6 × 3
    ##    fold metric value
    ##   <int> <chr>  <dbl>
    ## 1     1 cindex 0.532
    ## 2     1 ibs    0.242
    ## 3     2 cindex 0.597
    ## 4     2 ibs    0.222
    ## 5     3 cindex 0.419
    ## 6     3 ibs    0.263

------------------------------------------------------------------------

## Hyperparameter Tuning

``` r
grid <- list(
  hidden     = list(c(16), c(32, 16)),
  lr         = c(1e-3),
  activation = c("relu"),
  epochs     = c(100),
  .loss_fn   = list(cox_loss, aft_loss),
  loss_name  = c("cox_loss", "aft_loss")
  )

tune_res <- tune_survdnn(
  formula = Surv(time, status) ~ age + karno + celltype,
  data = veteran,
  times = c(90, 300),
  metrics = c("cindex"),
  param_grid = grid,
  folds = 3,
  return = "summary"
  )
```

    ## Epoch 50 - Loss: 3.511837
    ## Epoch 100 - Loss: 3.387282
    ## Epoch 50 - Loss: 3.446851
    ## Epoch 100 - Loss: 3.280030
    ## Epoch 50 - Loss: 3.468261
    ## Epoch 100 - Loss: 3.449604
    ## Epoch 50 - Loss: 3.408476
    ## Epoch 100 - Loss: 3.277228
    ## Epoch 50 - Loss: 3.419980
    ## Epoch 100 - Loss: 3.368773
    ## Epoch 50 - Loss: 3.468292
    ## Epoch 100 - Loss: 3.403583
    ## Epoch 50 - Loss: 13.004675
    ## Epoch 100 - Loss: 10.427486
    ## Epoch 50 - Loss: 17.567770
    ## Epoch 100 - Loss: 13.712244
    ## Epoch 50 - Loss: 16.859138
    ## Epoch 100 - Loss: 12.991044
    ## Epoch 50 - Loss: 14.365346
    ## Epoch 100 - Loss: 11.264277
    ## Epoch 50 - Loss: 12.999846
    ## Epoch 100 - Loss: 10.277443
    ## Epoch 50 - Loss: 14.711557
    ## Epoch 100 - Loss: 11.296858
    ## Epoch 50 - Loss: 3.346302
    ## Epoch 100 - Loss: 3.314769
    ## Epoch 50 - Loss: 3.381723
    ## Epoch 100 - Loss: 3.258752
    ## Epoch 50 - Loss: 3.423177
    ## Epoch 100 - Loss: 3.371849
    ## Epoch 50 - Loss: 3.366550
    ## Epoch 100 - Loss: 3.219291
    ## Epoch 50 - Loss: 3.463642
    ## Epoch 100 - Loss: 3.426658
    ## Epoch 50 - Loss: 3.366479
    ## Epoch 100 - Loss: 3.451038
    ## Epoch 50 - Loss: 17.730415
    ## Epoch 100 - Loss: 14.149435
    ## Epoch 50 - Loss: 14.713129
    ## Epoch 100 - Loss: 10.928744
    ## Epoch 50 - Loss: 12.516415
    ## Epoch 100 - Loss: 9.061637
    ## Epoch 50 - Loss: 13.635848
    ## Epoch 100 - Loss: 9.821770
    ## Epoch 50 - Loss: 12.190846
    ## Epoch 100 - Loss: 9.013781
    ## Epoch 50 - Loss: 15.864221
    ## Epoch 100 - Loss: 12.559014

``` r
print(tune_res)
```

    ## # A tibble: 4 × 8
    ##   hidden       lr activation epochs loss_name metric  mean     sd
    ##   <list>    <dbl> <chr>       <dbl> <chr>     <chr>  <dbl>  <dbl>
    ## 1 <dbl [2]> 0.001 relu          100 cox_loss  cindex 0.701 0.0567
    ## 2 <dbl [2]> 0.001 relu          100 aft_loss  cindex 0.692 0.0298
    ## 3 <dbl [1]> 0.001 relu          100 cox_loss  cindex 0.654 0.0804
    ## 4 <dbl [1]> 0.001 relu          100 aft_loss  cindex 0.596 0.164

------------------------------------------------------------------------

## Plot Survival Curves

``` r
plot(mod1, group_by = "celltype", times = 1:300)  # individual + mean curves
```

![](README_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

``` r
plot(mod1, group_by = "celltype", times = 1:300, plot_mean_only = TRUE)  # mean only
```

![](README_files/figure-gfm/unnamed-chunk-6-2.png)<!-- -->

------------------------------------------------------------------------

## Documentation

Full documentation with examples, loss definitions, metric evaluators,
and model structure is available via:

``` r
help(package = "survdnn")
?survdnn
?cox_loss
?tune_survdnn
?plot.survdnn
```

------------------------------------------------------------------------

## Testing

Run all unit tests (TODO):

``` r
devtools::test()
```

------------------------------------------------------------------------

## Contributions

Contributions, issues, and feature requests are welcome. See the [Issues
page](https://github.com/ielbadisy/survdnn/issues).

------------------------------------------------------------------------

## License

MIT © [Imad El Badisy](mailto:elbadisyimad@gmail.com)
