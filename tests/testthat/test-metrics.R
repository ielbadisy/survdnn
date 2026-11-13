test_that("cindex_survmat computes a valid C-index", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  mod <- survdnn(Surv(time, status) ~ age + karno + celltype,
                 data = data, epochs = 5, verbose = FALSE)

  times <- c(30, 90, 180)
  sp <- predict(mod, newdata = data, times = times, type = "survival")
  y <- model.response(model.frame(mod$formula, data))

  cidx <- cindex_survmat(y, sp, t_star = 90)
  expect_type(cidx, "double")
  expect_true(cidx >= 0 && cidx <= 1, info = paste("C-index out of bounds:", cidx))
})


test_that("brier returns a valid score", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  mod <- survdnn(Surv(time, status) ~ age + karno + celltype,
                 data = data, epochs = 5, verbose = FALSE)

  times <- 90
  sp <- predict(mod, newdata = data, times = times, type = "survival")
  y <- model.response(model.frame(mod$formula, data))

  bs <- brier(y, pre_sp = sp[["t=90"]], t_star = 90)
  expect_type(bs, "double")
  expect_true(bs >= 0 && bs <= 1, info = paste("Brier score out of bounds:", bs))
})


test_that("ibs_survmat returns a valid integrated Brier score", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  mod <- survdnn(Surv(time, status) ~ age + karno + celltype,
                 data = data, epochs = 5, verbose = FALSE)

  times <- c(30, 90, 180)
  sp <- predict(mod, newdata = data, times = times, type = "survival")
  y <- model.response(model.frame(mod$formula, data))

  ibs <- ibs_survmat(y, sp, times)
  expect_type(ibs, "double")
  expect_true(ibs >= 0 && ibs <= 1, info = paste("IBS out of bounds:", ibs))
})


test_that("metric functions handle dimension mismatches gracefully", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  y <- Surv(data$time, data$status)
  wrong_sp <- matrix(runif(200), nrow = 100)  # should be 137 rows

  expect_error(
    ibs_survmat(y, wrong_sp, times = c(30, 60)),
    regexp = "Length of predictions"
  )

  expect_error(
    brier(y, pre_sp = runif(100), t_star = 60),
    regexp = "Length of predictions"
  )
})
