test_that("plot.survdnn returns a ggplot object (default)", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  mod <- survdnn(Surv(time, status) ~ age + karno + celltype,
                 data = data, hidden = c(8), epochs = 5, verbose = FALSE)

  p <- plot(mod)

  expect_s3_class(p, "ggplot")
})

test_that("plot.survdnn supports group_by and plot_mean_only", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  mod <- survdnn(Surv(time, status) ~ age + karno + celltype,
                 data = data, hidden = c(8), epochs = 5, verbose = FALSE)

  # grouped + mean only
  p1 <- plot(mod, group_by = "celltype", plot_mean_only = TRUE, times = 1:100)
  expect_s3_class(p1, "ggplot")

  # grouped + individual + mean
  p2 <- plot(mod, group_by = "celltype", plot_mean_only = FALSE, add_mean = TRUE, times = 1:100)
  expect_s3_class(p2, "ggplot")
})

test_that("plot.survdnn fails with wrong inputs", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  expect_error(survdnn:::plot.survdnn("not a model"), "inherits")
})


