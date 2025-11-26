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
    c("model", "formula", "data", "xnames", "x_center", "x_scale",
      "loss_history", "final_loss", "loss", "activation", "hidden", "lr", "epochs"),
    ignore.order = TRUE
  )
})
