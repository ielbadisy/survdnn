test_that("survdnn accepts multiple optimizers", {
  skip_if_not_installed("torch")
  skip_if_not(torch::torch_is_installed())

  veteran <- survival::veteran

  opts <- c("adam", "adamw", "sgd", "rmsprop", "adagrad")

  for (opt in opts) {
    mod <- survdnn(
      Surv(time, status) ~ age + karno + celltype,
      data       = veteran,
      hidden     = c(8L, 4L),
      activation = "relu",
      lr         = 1e-3,
      epochs     = 3L,
      loss       = "cox",
      optimizer  = opt,
      verbose    = FALSE,
      .device    = "cpu"
    )

    expect_s3_class(mod, "survdnn")
    expect_equal(mod$optimizer, opt)
    expect_true(length(mod$loss_history) >= 1L)
  }
})

test_that("optim_args is passed to optimizer", {
  skip_if_not_installed("torch")
  skip_if_not(torch::torch_is_installed())

  veteran <- survival::veteran

  mod <- survdnn(
    Surv(time, status) ~ age + karno + celltype,
    data       = veteran,
    hidden     = c(8L, 4L),
    activation = "relu",
    lr         = 1e-3,
    epochs     = 3L,
    loss       = "cox",
    optimizer  = "sgd",
    optim_args = list(momentum = 0.9),
    verbose    = FALSE,
    .device    = "cpu"
  )

  expect_s3_class(mod, "survdnn")
  expect_equal(mod$optimizer, "sgd")
})
