test_that("tune_survdnn returns correct structure for all modes", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  times <- c(90, 300)
  grid <- list(
    hidden     = list(c(8)),
    lr         = c(1e-3),
    activation = c("relu"),
    epochs     = c(5),
    .loss_fn   = list(cox_loss),
    loss_name  = c("cox_loss")
  )

  res_all <- tune_survdnn(
    formula = Surv(time, status) ~ age + karno + celltype,
    data = data,
    times = times,
    metrics = "cindex",
    param_grid = grid,
    folds = 2,
    .seed = 42,
    refit = FALSE,
    return = "all"
  )

  expect_s3_class(res_all, "data.frame")
  expect_true(all(c("metric", "value", "loss_name") %in% names(res_all)))

  res_summary <- tune_survdnn(
    formula = Surv(time, status) ~ age + karno + celltype,
    data = data,
    times = times,
    metrics = "cindex",
    param_grid = grid,
    folds = 2,
    .seed = 42,
    refit = FALSE,
    return = "summary"
  )

  expect_s3_class(res_summary, "data.frame")
  expect_true(all(c("metric", "mean", "sd") %in% names(res_summary)))

  res_best <- tune_survdnn(
    formula = Surv(time, status) ~ age + karno + celltype,
    data = data,
    times = times,
    metrics = "cindex",
    param_grid = grid,
    folds = 2,
    .seed = 42,
    refit = FALSE,
    return = "best_model"
  )

  expect_s3_class(res_best, "data.frame")
  expect_true(all(c("hidden", "lr", "activation", "epochs", ".loss_fn", "loss_name") %in% names(res_best)))
})

test_that("tune_survdnn works with refit = TRUE and returns survdnn model", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  times <- c(90, 300)
  grid <- list(
    hidden     = list(c(8)),
    lr         = c(1e-3),
    activation = c("relu"),
    epochs     = c(5),
    .loss_fn   = list(cox_loss),
    loss_name  = c("cox_loss")
  )

  mod <- tune_survdnn(
    formula = Surv(time, status) ~ age + karno + celltype,
    data = data,
    times = times,
    metrics = "cindex",
    param_grid = grid,
    folds = 2,
    .seed = 123,
    refit = TRUE,
    return = "best_model"
  )

  expect_s3_class(mod, "survdnn")
  expect_true(inherits(mod$model, "nn_module"))
})

test_that("summarize_tune_survdnn aggregates correctly and throws on bad input", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  times <- c(90, 300)
  grid <- list(
    hidden     = list(c(8)),
    lr         = c(1e-3),
    activation = c("relu"),
    epochs     = c(5),
    .loss_fn   = list(cox_loss),
    loss_name  = c("cox_loss")
  )

  res_all <- tune_survdnn(
    formula = Surv(time, status) ~ age + karno + celltype,
    data = data,
    times = times,
    metrics = "cindex",
    param_grid = grid,
    folds = 2,
    .seed = 42,
    return = "all"
  )

  smry <- summarize_tune_survdnn(res_all)
  expect_s3_class(smry, "data.frame")
  expect_true(all(c("metric", "mean", "sd") %in% names(smry)))

  expect_error(summarize_tune_survdnn(data.frame(a = 1, b = 2)), "tune_survdnn")
})
