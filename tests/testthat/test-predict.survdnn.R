test_that("predict.survdnn works for type = 'lp'", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  mod <- survdnn(Surv(time, status) ~ age + karno + celltype,
                 data = data, epochs = 10, verbose = FALSE)

  lp <- predict(mod, newdata = data, type = "lp")

  expect_type(lp, "double")
  expect_length(lp, nrow(data))
  expect_false(anyNA(lp))
})

test_that("predict.survdnn works for type = 'survival'", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  times <- c(30, 90, 180)
  mod <- survdnn(Surv(time, status) ~ age + karno + celltype,
                 data = data, epochs = 10, verbose = FALSE)

  surv <- predict(mod, newdata = data, type = "survival", times = times)

  expect_s3_class(surv, "data.frame")
  expect_equal(ncol(surv), length(times))
  expect_equal(nrow(surv), nrow(data))
  expect_true(all(surv >= 0 & surv <= 1))
  expect_named(surv, paste0("t=", times))
})

test_that("predict.survdnn works for type = 'risk'", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  mod <- survdnn(Surv(time, status) ~ age + karno + celltype,
                 data = data, epochs = 10, verbose = FALSE)

  risk <- predict(mod, newdata = data, type = "risk", times = 180)

  expect_type(risk, "double")
  expect_length(risk, nrow(data))
  expect_true(all(risk >= 0 & risk <= 1))
})

test_that("predict.survdnn handles invalid or missing times properly", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  mod <- survdnn(Surv(time, status) ~ age + karno + celltype,
                 data = data, epochs = 10, verbose = FALSE)

  expect_error(predict(mod, newdata = data, type = "survival"),
               "`times` must be specified")

  expect_error(predict(mod, newdata = data, type = "risk", times = c(30, 60)),
               "For type = 'risk', `times` must be a single value.")
})
