test_that("print.survdnn prints without error and returns invisibly", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  mod <- survdnn(Surv(time, status) ~ age + karno + celltype,
                 data = data, epochs = 5, verbose = FALSE)

  expect_invisible(print(mod))
  expect_s3_class(mod, "survdnn")
})
