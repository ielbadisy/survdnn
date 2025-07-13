test_that("summary.survdnn prints key summary components", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  mod <- survdnn(Surv(time, status) ~ age + karno + celltype,
                 data = data, epochs = 5, verbose = FALSE)

  output <- capture.output(summary(mod))

  expect_true(any(grepl("Model architecture", output)))
  expect_true(any(grepl("Data summary", output)))
  expect_true(any(grepl("Event rate", output)))

})
