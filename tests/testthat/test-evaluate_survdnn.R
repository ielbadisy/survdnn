test_that("evaluatesurvdnn returns expected metric values", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  mod <- survdnn(Surv(time, status) ~ age + karno + celltype,
                 data = data, epochs = 5, verbose = FALSE)

  times <- c(30, 90, 180)

  out <- evaluatesurvdnn(mod, metrics = c("cindex", "ibs", "brier"), times = times)

  expect_s3_class(out, "data.frame")
  expect_true(all(c("metric", "value") %in% names(out)))
  expect_true(any(out$metric == "cindex"))
  expect_true(any(out$metric == "ibs"))
  expect_true(any(out$metric == "brier"))
})

test_that("evaluatesurvdnn errors with unknown metrics or missing times", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  mod <- survdnn(Surv(time, status) ~ age + karno + celltype,
                 data = data, epochs = 5, verbose = FALSE)

  expect_error(evaluatesurvdnn(mod, metrics = "abc", times = 100), "Unknown metric")
  expect_error(evaluatesurvdnn(mod, metrics = "cindex"), "must provide `times`")
})
