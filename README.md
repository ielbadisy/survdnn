
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
  - `"cox_l2"`: penalized Cox
  - `"aft"`: Accelerated Failure Time
  - `"coxtime"`: deep time-dependent Cox (like DeepSurv)
- Evaluation: C-index, Brier score, Integrated Brier Score (IBS)
- Model selection with `cv_survdnn()` and `tune_survdnn()`
- Prediction of survival curves via `predict()` and `plot()`

------------------------------------------------------------------------

## Installation

``` r
# Install from GitHub
# install.packages("remotes")
remotes::install_github("ielbadisy/survdnn")

# Or clone and install locally
# git clone https://github.com/ielbadisy/survdnn.git
# setwd("survdnn")
# devtools::install()
```

------------------------------------------------------------------------

## Quick Example

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
```

    ## Epoch 50 - Loss: 4.117886
    ## Epoch 100 - Loss: 4.023495
    ## Epoch 150 - Loss: 3.965827
    ## Epoch 200 - Loss: 3.955929
    ## Epoch 250 - Loss: 3.905104
    ## Epoch 300 - Loss: 3.850240

``` r
summary(mod)
```

    ## 

    ## ── Summary of survdnn model ────────────────────────────────

    ## 
    ## Formula:
    ##   Surv(time, status) ~ age + karno + celltype
    ## <environment: 0x625b8ae3d7d0>
    ## 
    ## Model architecture:
    ##   Hidden layers:  32 : 16 
    ##   Activation:  relu 
    ##   Dropout:  0.3 
    ##   Final loss:  3.850240 
    ## 
    ## Training summary:
    ##   Epochs:  300 
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

![](README_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

------------------------------------------------------------------------

## Loss Functions

``` r
# Cox partial likelihood
mod1 <- survdnn(
  Surv(time, status) ~ age + karno,
  data = veteran,
  loss = "cox",
  epochs = 200
  )
```

    ## Epoch 50 - Loss: 3.976199
    ## Epoch 100 - Loss: 3.910449
    ## Epoch 150 - Loss: 3.969741
    ## Epoch 200 - Loss: 3.858460

``` r
# Accelerated Failure Time
mod2 <- survdnn(
  Surv(time, status) ~ age + karno,
  data = veteran,
  loss = "aft",
  epochs = 300
  )
```

    ## Epoch 50 - Loss: 14.876557
    ## Epoch 100 - Loss: 14.589748
    ## Epoch 150 - Loss: 14.596359
    ## Epoch 200 - Loss: 13.964574
    ## Epoch 250 - Loss: 13.534356
    ## Epoch 300 - Loss: 13.319471

``` r
# Deep time-dependent Cox (Coxtime)
mod3 <- survdnn(
  Surv(time, status) ~ age + karno,
  data = veteran,
  loss = "coxtime",
  epochs = 100
  )
```

    ## Epoch 50 - Loss: 4.805439
    ## Epoch 100 - Loss: 4.767772

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
  epochs = 300
)
```

    ## Epoch 50 - Loss: 3.449679
    ## Epoch 100 - Loss: 3.532599
    ## Epoch 150 - Loss: 3.468612
    ## Epoch 200 - Loss: 3.457351
    ## Epoch 250 - Loss: 3.455626
    ## Epoch 300 - Loss: 3.435783
    ## Epoch 50 - Loss: 3.691198
    ## Epoch 100 - Loss: 3.505430
    ## Epoch 150 - Loss: 3.564598
    ## Epoch 200 - Loss: 3.569710
    ## Epoch 250 - Loss: 3.605495
    ## Epoch 300 - Loss: 3.520325
    ## Epoch 50 - Loss: 3.579567
    ## Epoch 100 - Loss: 3.519223
    ## Epoch 150 - Loss: 3.549295
    ## Epoch 200 - Loss: 3.503947
    ## Epoch 250 - Loss: 3.534834
    ## Epoch 300 - Loss: 3.451947

``` r
print(cv_results)
```

    ## # A tibble: 6 × 3
    ##    fold metric value
    ##   <int> <chr>  <dbl>
    ## 1     1 cindex 0.719
    ## 2     1 ibs    0.217
    ## 3     2 cindex 0.591
    ## 4     2 ibs    0.234
    ## 5     3 cindex 0.686
    ## 6     3 ibs    0.245

------------------------------------------------------------------------

## Hyperparameter Tuning

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
```

    ## Epoch 50 - Loss: 14.334056
    ## Epoch 100 - Loss: 10.385813
    ## Epoch 50 - Loss: 13.184210
    ## Epoch 100 - Loss: 9.531919
    ## Epoch 50 - Loss: 15.402397
    ## Epoch 100 - Loss: 11.555803
    ## Epoch 50 - Loss: 3.408515
    ## Epoch 100 - Loss: 3.343168
    ## Epoch 50 - Loss: 3.489057
    ## Epoch 100 - Loss: 3.404249
    ## Epoch 50 - Loss: 3.557032
    ## Epoch 100 - Loss: 3.439668
    ## Epoch 50 - Loss: 4.277906
    ## Epoch 100 - Loss: 4.209718
    ## Epoch 50 - Loss: 4.256487
    ## Epoch 100 - Loss: 4.162229
    ## Epoch 50 - Loss: 4.352544
    ## Epoch 100 - Loss: 4.278419
    ## Epoch 50 - Loss: 13.616202
    ## Epoch 100 - Loss: 10.354403
    ## Epoch 150 - Loss: 7.703625
    ## Epoch 200 - Loss: 5.299365
    ## Epoch 250 - Loss: 3.340566
    ## Epoch 300 - Loss: 2.864841
    ## Epoch 50 - Loss: 19.126936
    ## Epoch 100 - Loss: 15.268164
    ## Epoch 150 - Loss: 11.585573
    ## Epoch 200 - Loss: 8.821329
    ## Epoch 250 - Loss: 6.323870
    ## Epoch 300 - Loss: 3.819921
    ## Epoch 50 - Loss: 13.018722
    ## Epoch 100 - Loss: 10.276742
    ## Epoch 150 - Loss: 6.673223
    ## Epoch 200 - Loss: 4.137823
    ## Epoch 250 - Loss: 2.929209
    ## Epoch 300 - Loss: 2.631979
    ## Epoch 50 - Loss: 3.604008
    ## Epoch 100 - Loss: 3.428994
    ## Epoch 150 - Loss: 3.377207
    ## Epoch 200 - Loss: 3.331337
    ## Epoch 250 - Loss: 3.324245
    ## Epoch 300 - Loss: 3.291971
    ## Epoch 50 - Loss: 3.421722
    ## Epoch 100 - Loss: 3.308124
    ## Epoch 150 - Loss: 3.275615
    ## Epoch 200 - Loss: 3.260485
    ## Epoch 250 - Loss: 3.282410
    ## Epoch 300 - Loss: 3.199148
    ## Epoch 50 - Loss: 3.505374
    ## Epoch 100 - Loss: 3.455868
    ## Epoch 150 - Loss: 3.435581
    ## Epoch 200 - Loss: 3.336907
    ## Epoch 250 - Loss: 3.348224
    ## Epoch 300 - Loss: 3.356287
    ## Epoch 50 - Loss: 4.395680
    ## Epoch 100 - Loss: 4.303700
    ## Epoch 150 - Loss: 4.211801
    ## Epoch 200 - Loss: 4.182858
    ## Epoch 250 - Loss: 4.146912
    ## Epoch 300 - Loss: 4.131495
    ## Epoch 50 - Loss: 4.322754
    ## Epoch 100 - Loss: 4.248060
    ## Epoch 150 - Loss: 4.159159
    ## Epoch 200 - Loss: 4.116855
    ## Epoch 250 - Loss: 4.103096
    ## Epoch 300 - Loss: 4.026472
    ## Epoch 50 - Loss: 4.405371
    ## Epoch 100 - Loss: 4.346283
    ## Epoch 150 - Loss: 4.298858
    ## Epoch 200 - Loss: 4.213375
    ## Epoch 250 - Loss: 4.179698
    ## Epoch 300 - Loss: 4.180149
    ## Epoch 50 - Loss: 13.227220
    ## Epoch 100 - Loss: 9.641831
    ## Epoch 50 - Loss: 11.296117
    ## Epoch 100 - Loss: 7.826581
    ## Epoch 50 - Loss: 13.598725
    ## Epoch 100 - Loss: 10.054354
    ## Epoch 50 - Loss: 3.414766
    ## Epoch 100 - Loss: 3.308276
    ## Epoch 50 - Loss: 3.406729
    ## Epoch 100 - Loss: 3.416101
    ## Epoch 50 - Loss: 3.374182
    ## Epoch 100 - Loss: 3.377437
    ## Epoch 50 - Loss: 4.174135
    ## Epoch 100 - Loss: 4.074947
    ## Epoch 50 - Loss: 4.172358
    ## Epoch 100 - Loss: 4.115020
    ## Epoch 50 - Loss: 4.252332
    ## Epoch 100 - Loss: 4.120183
    ## Epoch 50 - Loss: 12.136313
    ## Epoch 100 - Loss: 9.386659
    ## Epoch 150 - Loss: 6.603240
    ## Epoch 200 - Loss: 5.046573
    ## Epoch 250 - Loss: 3.408053
    ## Epoch 300 - Loss: 2.594421
    ## Epoch 50 - Loss: 15.169104
    ## Epoch 100 - Loss: 11.791401
    ## Epoch 150 - Loss: 8.789195
    ## Epoch 200 - Loss: 6.137484
    ## Epoch 250 - Loss: 3.864896
    ## Epoch 300 - Loss: 3.024428
    ## Epoch 50 - Loss: 15.911388
    ## Epoch 100 - Loss: 12.432424
    ## Epoch 150 - Loss: 9.772544
    ## Epoch 200 - Loss: 7.005900
    ## Epoch 250 - Loss: 4.463702
    ## Epoch 300 - Loss: 3.470662
    ## Epoch 50 - Loss: 3.440627
    ## Epoch 100 - Loss: 3.248658
    ## Epoch 150 - Loss: 3.225950
    ## Epoch 200 - Loss: 3.193726
    ## Epoch 250 - Loss: 3.040551
    ## Epoch 300 - Loss: 3.119142
    ## Epoch 50 - Loss: 3.447996
    ## Epoch 100 - Loss: 3.384627
    ## Epoch 150 - Loss: 3.318794
    ## Epoch 200 - Loss: 3.301319
    ## Epoch 250 - Loss: 3.286160
    ## Epoch 300 - Loss: 3.336246
    ## Epoch 50 - Loss: 3.384303
    ## Epoch 100 - Loss: 3.381250
    ## Epoch 150 - Loss: 3.303163
    ## Epoch 200 - Loss: 3.284627
    ## Epoch 250 - Loss: 3.286431
    ## Epoch 300 - Loss: 3.329447
    ## Epoch 50 - Loss: 4.383369
    ## Epoch 100 - Loss: 4.198235
    ## Epoch 150 - Loss: 4.169467
    ## Epoch 200 - Loss: 4.142875
    ## Epoch 250 - Loss: 4.100346
    ## Epoch 300 - Loss: 4.102182
    ## Epoch 50 - Loss: 4.220581
    ## Epoch 100 - Loss: 4.080726
    ## Epoch 150 - Loss: 4.053489
    ## Epoch 200 - Loss: 4.009274
    ## Epoch 250 - Loss: 4.048024
    ## Epoch 300 - Loss: 3.978660
    ## Epoch 50 - Loss: 4.278310
    ## Epoch 100 - Loss: 4.138876
    ## Epoch 150 - Loss: 4.081411
    ## Epoch 200 - Loss: 4.074841
    ## Epoch 250 - Loss: 4.031276
    ## Epoch 300 - Loss: 4.042162

``` r
print(tune_res)
```

    ## # A tibble: 12 × 8
    ##    hidden       lr activation epochs loss    metric  mean     sd
    ##    <list>    <dbl> <chr>       <dbl> <chr>   <chr>  <dbl>  <dbl>
    ##  1 <dbl [2]> 0.001 relu          100 cox     cindex 0.729 0.0198
    ##  2 <dbl [1]> 0.001 relu          100 cox     cindex 0.719 0.0655
    ##  3 <dbl [1]> 0.001 relu          300 cox     cindex 0.712 0.0257
    ##  4 <dbl [1]> 0.001 relu          100 coxtime cindex 0.711 0.0486
    ##  5 <dbl [2]> 0.001 relu          300 cox     cindex 0.710 0.0613
    ##  6 <dbl [1]> 0.001 relu          300 coxtime cindex 0.697 0.0478
    ##  7 <dbl [2]> 0.001 relu          300 aft     cindex 0.695 0.0163
    ##  8 <dbl [1]> 0.001 relu          300 aft     cindex 0.679 0.0296
    ##  9 <dbl [2]> 0.001 relu          100 aft     cindex 0.678 0.0255
    ## 10 <dbl [2]> 0.001 relu          300 coxtime cindex 0.673 0.114 
    ## 11 <dbl [2]> 0.001 relu          100 coxtime cindex 0.650 0.0652
    ## 12 <dbl [1]> 0.001 relu          100 aft     cindex 0.544 0.131

------------------------------------------------------------------------

## Plot Survival Curves

``` r
plot(mod1, group_by = "celltype", times = 1:300)
```

![](README_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

``` r
plot(mod1, group_by = "celltype", times = 1:300, plot_mean_only = TRUE)
```

![](README_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

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

## Availability

The `survdnn` R package is available at:
<https://github.com/ielbadisy/survdnn>

The package is currently under submission to CRAN.

------------------------------------------------------------------------

## Contributions

Contributions, issues, and feature requests are welcome. Open an
[issue](https://github.com/ielbadisy/survdnn/issues) or submit a pull
request!

------------------------------------------------------------------------

## License

MIT © [Imad El Badisy](mailto:elbadisyimad@gmail.com)
