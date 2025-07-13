test_that("survdnn() fits a model and returns correct structure", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  set.seed(123)
  n <- 100
  df <- data.frame(
    time = rexp(n, rate = 0.1),
    status = rbinom(n, 1, 0.7),
    x1 = rnorm(n),
    x2 = rbinom(n, 1, 0.5)
  )

  mod <- survdnn(
    Surv(time, status) ~ x1 + x2,
    data = df,
    hidden = c(8, 4),
    activation = "relu",
    epochs = 10,
    lr = 1e-3,
    .loss_fn = cox_loss,
    verbose = FALSE
  )

  expect_s3_class(mod, "survdnn")
  expect_true(inherits(mod$model, "nn_module"))
  expect_equal(deparse(mod$formula), deparse(Surv(time, status) ~ x1 + x2))
  expect_type(mod$loss, "double")
  expect_length(mod$loss_history, 10)
  expect_named(mod, c(
    "model", "formula", "data", "xnames", "x_center", "x_scale",
    "loss", "loss_history", "activation", "hidden", "lr", "epochs",
    ".loss_fn", "loss_name"
  ))
})

test_that("build_dnn builds a valid nn_sequential model", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  net <- build_dnn(input_dim = 10, hidden = c(16, 8), activation = "relu")

  expect_true(inherits(net, "nn_sequential"))
  expect_length(net, length(c(16, 8)) * 4 + 1)  # dynamic
  expect_true(all(sapply(net$parameters, function(p) inherits(p, "torch_tensor"))))
})

test_that("survdnn() throws error on invalid input", {
  dummy_data <- data.frame(
    time = rexp(10),
    status = rbinom(10, 1, 0.5),
    x = rnorm(10)
  )

  expect_error(survdnn("not a formula", dummy_data), "inherits\\(formula, \"formula\"\\)")
  expect_error(survdnn(Surv(time, status) ~ x, list()), "is.data.frame")
  expect_error(survdnn(Surv(time, status) ~ x, dummy_data, .loss_fn = "notafn"), "must be a function")
})
