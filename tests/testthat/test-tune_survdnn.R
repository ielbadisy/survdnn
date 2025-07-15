test_that("tune_survdnn returns correct structure for all modes", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  param_grid <- list(
    hidden     = list(c(8), c(8, 4)),
    lr         = c(1e-3),
    activation = c("relu"),
    epochs     = c(5),
    loss       = c("cox")
  )
  times <- c(30, 90)

  all_res <- tune_survdnn(
    Surv(time, status) ~ age + karno + celltype,
    data = data,
    times = times,
    metrics = "cindex",
    param_grid = param_grid,
    folds = 2,
    .seed = 123,
    refit = FALSE,
    return = "all"
  )

  expect_s3_class(all_res, "data.frame")
  expect_true(all(c("fold", "metric", "value") %in% names(all_res)))

  summary_res <- tune_survdnn(
    Surv(time, status) ~ age + karno + celltype,
    data = data,
    times = times,
    metrics = "cindex",
    param_grid = param_grid,
    folds = 2,
    .seed = 123,
    refit = FALSE,
    return = "summary"
  )

  expect_s3_class(summary_res, "data.frame")
  expect_true(all(c("metric", "mean", "sd") %in% names(summary_res)))

  best_cfg <- tune_survdnn(
    Surv(time, status) ~ age + karno + celltype,
    data = data,
    times = times,
    metrics = "cindex",
    param_grid = param_grid,
    folds = 2,
    .seed = 123,
    refit = FALSE,
    return = "best_model"
  )

  expect_s3_class(best_cfg, "data.frame")
  expect_true(all(c("hidden", "lr", "activation", "epochs", "loss") %in% names(best_cfg)))
})

test_that("tune_survdnn works with refit = TRUE and returns survdnn model", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  param_grid <- list(
    hidden     = list(c(8)),
    lr         = c(1e-3),
    activation = c("relu"),
    epochs     = c(3),
    loss       = c("cox")
  )
  times <- c(30, 90)

  mod <- tune_survdnn(
    Surv(time, status) ~ age + karno + celltype,
    data = data,
    times = times,
    metrics = "cindex",
    param_grid = param_grid,
    folds = 2,
    .seed = 42,
    refit = TRUE,
    return = "best_model"
  )

  expect_s3_class(mod, "survdnn")
  expect_s3_class(mod$model, "nn_module")
})

test_that("summarize_tune_survdnn aggregates correctly and throws on bad input", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  param_grid <- list(
    hidden     = list(c(8)),
    lr         = c(1e-3),
    activation = c("relu"),
    epochs     = c(3),
    loss       = c("cox")
  )
  times <- c(30, 90)

  all_res <- tune_survdnn(
    Surv(time, status) ~ age + karno + celltype,
    data = data,
    times = times,
    metrics = "brier",  
    param_grid = param_grid,
    folds = 2,
    .seed = 123,
    refit = FALSE,
    return = "all"
  )

  sm <- summarize_tune_survdnn(all_res, by_time = TRUE)
  expect_s3_class(sm, "data.frame")

  if ("time" %in% names(all_res)) {
    expect_true("time" %in% names(sm))
  }

  expect_error(summarize_tune_survdnn(data.frame(a = 1)), "Input must be the result")
})
