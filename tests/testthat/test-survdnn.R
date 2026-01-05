test_that("survdnn() fits a model and returns correct structure", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  set.seed(123)
  n <- 100
  df <- data.frame(
    time   = rexp(n, rate = 0.1),
    status = rbinom(n, 1, 0.7),
    x1     = rnorm(n),
    x2     = rbinom(n, 1, 0.5)
  )

  mod <- survdnn(
    Surv(time, status) ~ x1 + x2,
    data = df,
    hidden = c(8, 4),
    activation = "relu",
    epochs = 10,
    lr = 1e-3,
    loss = "cox",
    verbose = FALSE
  )

  expect_s3_class(mod, "survdnn")
  expect_true(inherits(mod$model, "nn_module"))
  expect_equal(deparse(mod$formula), deparse(Surv(time, status) ~ x1 + x2))
  expect_type(mod$final_loss, "double")
  expect_length(mod$loss_history, 10)

  expect_named(
    mod,
    c(
      "activation",
      "aft_loc",
      "aft_log_sigma",
      "batch_norm",
      "coxtime_time_center",
      "coxtime_time_scale",
      "data",
      "device",
      "dropout",
      "epochs",
      "final_loss",
      "formula",
      "hidden",
      "loss",
      "loss_history",
      "lr",
      "model",
      "na_action",
      "optim_args",
      "optimizer",
      "x_center",
      "x_scale",
      "xnames"
    ),
    ignore.order = TRUE
  )

  # sanity: for loss="cox", AFT/CoxTime metadata should be NA
  expect_true(is.na(mod$aft_log_sigma))
  expect_true(is.na(mod$aft_loc))
  expect_true(is.na(mod$coxtime_time_center))
  expect_true(is.na(mod$coxtime_time_scale))
})


test_that("survdnn() is reproducible given .seed", {
  skip_on_cran()
  skip_if_not_installed("torch")
  skip_if_not(torch::torch_is_installed())

  set.seed(123)
  n <- 80
  df <- data.frame(
    time   = rexp(n, rate = 0.1),
    status = rbinom(n, 1, 0.7),
    x1     = rnorm(n),
    x2     = rbinom(n, 1, 0.5)
  )

  mod1 <- survdnn(
    Surv(time, status) ~ x1 + x2,
    data   = df,
    hidden = c(8),
    activation = "relu",
    lr     = 1e-3,
    epochs = 10,
    loss   = "cox",
    verbose = FALSE,
    .seed   = 999
  )

  mod2 <- survdnn(
    Surv(time, status) ~ x1 + x2,
    data   = df,
    hidden = c(8),
    activation = "relu",
    lr     = 1e-3,
    epochs = 10,
    loss   = "cox",
    verbose = FALSE,
    .seed   = 999
  )

  expect_equal(mod1$loss_history, mod2$loss_history)
})
