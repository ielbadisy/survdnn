test_that("plot_loss returns a ggplot object", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  veteran <- survival::veteran

  mod <- survdnn(
    Surv(time, status) ~ age + karno + celltype,
    data   = veteran,
    epochs = 5,
    loss   = "cox",
    verbose = FALSE,
    .device = "cpu"
  )

  p <- plot_loss(mod)
  expect_s3_class(p, "ggplot")
})
