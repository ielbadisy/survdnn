
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

    ## Epoch 50 - Loss: 4.017306
    ## Epoch 100 - Loss: 3.976948
    ## Epoch 150 - Loss: 3.904139
    ## Epoch 200 - Loss: 3.885131
    ## Epoch 250 - Loss: 3.860893
    ## Epoch 300 - Loss: 3.834231

``` r
summary(mod)
```

    ## 

    ## ── Summary of survdnn model ────────────────────────────────

    ## 
    ## Formula:
    ##   Surv(time, status) ~ age + karno + celltype
    ## <environment: 0x601199a5d900>
    ## 
    ## Model architecture:
    ##   Hidden layers:  32 : 16 
    ##   Activation:  relu 
    ##   Dropout:  0.3 
    ##   Final loss:  3.834231 
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

    ## Epoch 50 - Loss: 3.911210
    ## Epoch 100 - Loss: 3.883241
    ## Epoch 150 - Loss: 3.874349
    ## Epoch 200 - Loss: 3.812633

``` r
# Accelerated Failure Time
mod2 <- survdnn(
  Surv(time, status) ~ age + karno,
  data = veteran,
  loss = "aft",
  epochs = 300
  )
```

    ## Epoch 50 - Loss: 17.005127
    ## Epoch 100 - Loss: 16.583397
    ## Epoch 150 - Loss: 15.967889
    ## Epoch 200 - Loss: 15.324314
    ## Epoch 250 - Loss: 15.524298
    ## Epoch 300 - Loss: 14.680712

``` r
# Deep time-dependent Cox (Coxtime)
mod3 <- survdnn(
  Surv(time, status) ~ age + karno,
  data = veteran,
  loss = "coxtime",
  epochs = 100
  )
```

    ## Epoch 50 - Loss: 4.894759
    ## Epoch 100 - Loss: 4.836847

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

    ## Epoch 50 - Loss: 3.644217
    ## Epoch 100 - Loss: 3.637395
    ## Epoch 150 - Loss: 3.620273
    ## Epoch 200 - Loss: 3.581555
    ## Epoch 250 - Loss: 3.479215
    ## Epoch 300 - Loss: 3.569111
    ## Epoch 50 - Loss: 3.616415
    ## Epoch 100 - Loss: 3.533522
    ## Epoch 150 - Loss: 3.551425
    ## Epoch 200 - Loss: 3.595720
    ## Epoch 250 - Loss: 3.567378
    ## Epoch 300 - Loss: 3.519440
    ## Epoch 50 - Loss: 3.581872
    ## Epoch 100 - Loss: 3.628656
    ## Epoch 150 - Loss: 3.599951
    ## Epoch 200 - Loss: 3.506192
    ## Epoch 250 - Loss: 3.621629
    ## Epoch 300 - Loss: 3.483367

``` r
print(cv_results)
```

    ## # A tibble: 6 × 3
    ##    fold metric value
    ##   <int> <chr>  <dbl>
    ## 1     1 cindex 0.631
    ## 2     1 ibs    0.225
    ## 3     2 cindex 0.585
    ## 4     2 ibs    0.215
    ## 5     3 cindex 0.665
    ## 6     3 ibs    0.191

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

    ## Epoch 50 - Loss: 14.727397
    ## Epoch 100 - Loss: 11.155166
    ## Epoch 50 - Loss: 12.910063
    ## Epoch 100 - Loss: 9.430183
    ## Epoch 50 - Loss: 13.047706
    ## Epoch 100 - Loss: 10.300708
    ## Epoch 50 - Loss: 3.505261
    ## Epoch 100 - Loss: 3.368402
    ## Epoch 50 - Loss: 3.478866
    ## Epoch 100 - Loss: 3.420742
    ## Epoch 50 - Loss: 3.475290
    ## Epoch 100 - Loss: 3.459158
    ## Epoch 50 - Loss: 4.202677
    ## Epoch 100 - Loss: 4.167446
    ## Epoch 50 - Loss: 4.420293
    ## Epoch 100 - Loss: 4.350584
    ## Epoch 50 - Loss: 4.417453
    ## Epoch 100 - Loss: 4.304501
    ## Epoch 50 - Loss: 14.541389
    ## Epoch 100 - Loss: 10.945264
    ## Epoch 150 - Loss: 8.473359
    ## Epoch 200 - Loss: 5.348104
    ## Epoch 250 - Loss: 4.317372
    ## Epoch 300 - Loss: 3.172563
    ## Epoch 50 - Loss: 10.939583
    ## Epoch 100 - Loss: 7.925574
    ## Epoch 150 - Loss: 5.244921
    ## Epoch 200 - Loss: 3.448621
    ## Epoch 250 - Loss: 2.475841
    ## Epoch 300 - Loss: 1.948184
    ## Epoch 50 - Loss: 15.980975
    ## Epoch 100 - Loss: 12.818171
    ## Epoch 150 - Loss: 9.496651
    ## Epoch 200 - Loss: 6.914639
    ## Epoch 250 - Loss: 3.792582
    ## Epoch 300 - Loss: 3.371604
    ## Epoch 50 - Loss: 3.372404
    ## Epoch 100 - Loss: 3.362326
    ## Epoch 150 - Loss: 3.386676
    ## Epoch 200 - Loss: 3.335338
    ## Epoch 250 - Loss: 3.221668
    ## Epoch 300 - Loss: 3.375111
    ## Epoch 50 - Loss: 3.388513
    ## Epoch 100 - Loss: 3.313547
    ## Epoch 150 - Loss: 3.302353
    ## Epoch 200 - Loss: 3.282830
    ## Epoch 250 - Loss: 3.273268
    ## Epoch 300 - Loss: 3.289845
    ## Epoch 50 - Loss: 3.459574
    ## Epoch 100 - Loss: 3.397954
    ## Epoch 150 - Loss: 3.356992
    ## Epoch 200 - Loss: 3.395827
    ## Epoch 250 - Loss: 3.295613
    ## Epoch 300 - Loss: 3.381113
    ## Epoch 50 - Loss: 4.379249
    ## Epoch 100 - Loss: 4.278788
    ## Epoch 150 - Loss: 4.234133
    ## Epoch 200 - Loss: 4.236159
    ## Epoch 250 - Loss: 4.138520
    ## Epoch 300 - Loss: 4.119393
    ## Epoch 50 - Loss: 4.571877
    ## Epoch 100 - Loss: 4.436663
    ## Epoch 150 - Loss: 4.330639
    ## Epoch 200 - Loss: 4.276153
    ## Epoch 250 - Loss: 4.215656
    ## Epoch 300 - Loss: 4.166717
    ## Epoch 50 - Loss: 4.413413
    ## Epoch 100 - Loss: 4.376058
    ## Epoch 150 - Loss: 4.226744
    ## Epoch 200 - Loss: 4.158085
    ## Epoch 250 - Loss: 4.117296
    ## Epoch 300 - Loss: 4.089604
    ## Epoch 50 - Loss: 16.398582
    ## Epoch 100 - Loss: 12.791593
    ## Epoch 50 - Loss: 13.004353
    ## Epoch 100 - Loss: 9.463613
    ## Epoch 50 - Loss: 14.800050
    ## Epoch 100 - Loss: 12.186926
    ## Epoch 50 - Loss: 3.356088
    ## Epoch 100 - Loss: 3.246735
    ## Epoch 50 - Loss: 3.427070
    ## Epoch 100 - Loss: 3.348615
    ## Epoch 50 - Loss: 3.394627
    ## Epoch 100 - Loss: 3.346736
    ## Epoch 50 - Loss: 4.202281
    ## Epoch 100 - Loss: 4.121673
    ## Epoch 50 - Loss: 4.240664
    ## Epoch 100 - Loss: 4.142388
    ## Epoch 50 - Loss: 4.199209
    ## Epoch 100 - Loss: 4.113322
    ## Epoch 50 - Loss: 13.827906
    ## Epoch 100 - Loss: 10.583267
    ## Epoch 150 - Loss: 7.797911
    ## Epoch 200 - Loss: 5.747877
    ## Epoch 250 - Loss: 3.828708
    ## Epoch 300 - Loss: 2.490148
    ## Epoch 50 - Loss: 12.812648
    ## Epoch 100 - Loss: 9.808009
    ## Epoch 150 - Loss: 7.727947
    ## Epoch 200 - Loss: 4.772999
    ## Epoch 250 - Loss: 2.606256
    ## Epoch 300 - Loss: 2.158249
    ## Epoch 50 - Loss: 16.199535
    ## Epoch 100 - Loss: 13.163276
    ## Epoch 150 - Loss: 9.736091
    ## Epoch 200 - Loss: 7.939382
    ## Epoch 250 - Loss: 5.271013
    ## Epoch 300 - Loss: 3.729478
    ## Epoch 50 - Loss: 3.233447
    ## Epoch 100 - Loss: 3.245439
    ## Epoch 150 - Loss: 3.173748
    ## Epoch 200 - Loss: 3.103307
    ## Epoch 250 - Loss: 3.181788
    ## Epoch 300 - Loss: 3.040456
    ## Epoch 50 - Loss: 3.356755
    ## Epoch 100 - Loss: 3.319529
    ## Epoch 150 - Loss: 3.336454
    ## Epoch 200 - Loss: 3.356863
    ## Epoch 250 - Loss: 3.292748
    ## Epoch 300 - Loss: 3.273286
    ## Epoch 50 - Loss: 3.467583
    ## Epoch 100 - Loss: 3.245041
    ## Epoch 150 - Loss: 3.293297
    ## Epoch 200 - Loss: 3.385828
    ## Epoch 250 - Loss: 3.308768
    ## Epoch 300 - Loss: 3.283088
    ## Epoch 50 - Loss: 4.201644
    ## Epoch 100 - Loss: 4.079730
    ## Epoch 150 - Loss: 4.049956
    ## Epoch 200 - Loss: 4.041712
    ## Epoch 250 - Loss: 4.014086
    ## Epoch 300 - Loss: 3.999468
    ## Epoch 50 - Loss: 4.206083
    ## Epoch 100 - Loss: 4.121191
    ## Epoch 150 - Loss: 4.122781
    ## Epoch 200 - Loss: 4.012592
    ## Epoch 250 - Loss: 4.034368
    ## Epoch 300 - Loss: 3.980411
    ## Epoch 50 - Loss: 4.314287
    ## Epoch 100 - Loss: 4.190366
    ## Epoch 150 - Loss: 4.111125
    ## Epoch 200 - Loss: 4.084428
    ## Epoch 250 - Loss: 4.055722
    ## Epoch 300 - Loss: 4.067462

``` r
print(tune_res)
```

    ## # A tibble: 12 × 8
    ##    hidden       lr activation epochs loss    metric  mean     sd
    ##    <list>    <dbl> <chr>       <dbl> <chr>   <chr>  <dbl>  <dbl>
    ##  1 <dbl [1]> 0.001 relu          100 cox     cindex 0.729 0.0527
    ##  2 <dbl [2]> 0.001 relu          300 cox     cindex 0.714 0.0614
    ##  3 <dbl [2]> 0.001 relu          100 cox     cindex 0.712 0.0136
    ##  4 <dbl [1]> 0.001 relu          300 cox     cindex 0.711 0.0269
    ##  5 <dbl [2]> 0.001 relu          300 aft     cindex 0.699 0.0382
    ##  6 <dbl [2]> 0.001 relu          100 aft     cindex 0.694 0.0118
    ##  7 <dbl [1]> 0.001 relu          100 coxtime cindex 0.688 0.0843
    ##  8 <dbl [1]> 0.001 relu          300 coxtime cindex 0.688 0.0610
    ##  9 <dbl [1]> 0.001 relu          300 aft     cindex 0.686 0.0603
    ## 10 <dbl [2]> 0.001 relu          300 coxtime cindex 0.662 0.104 
    ## 11 <dbl [2]> 0.001 relu          100 coxtime cindex 0.628 0.0617
    ## 12 <dbl [1]> 0.001 relu          100 aft     cindex 0.609 0.0510

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
A stable version has been archived at Zenodo:
<https://doi.org/10.5281/zenodo.XXXXXXX>  
The package is currently under submission to CRAN.

------------------------------------------------------------------------

## Contributions

Contributions, issues, and feature requests are welcome. Open an
[issue](https://github.com/ielbadisy/survdnn/issues) or submit a pull
request!

------------------------------------------------------------------------

## License

MIT © [Imad El Badisy](mailto:elbadisyimad@gmail.com)
