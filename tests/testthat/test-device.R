test_that("survdnn_get_device falls back to CPU when torch is missing", {
  skip_if_not_installed("torch")
  skip_if_not(torch::torch_is_installed())

  dev_cpu <- survdnn_get_device("cpu")
  expect_s3_class(dev_cpu, "torch_device")
  expect_equal(dev_cpu$type, "cpu")
})

test_that("survdnn_get_device('auto') chooses CPU or CUDA consistently", {
  skip_if_not_installed("torch")
  skip_if_not(torch::torch_is_installed())

  dev <- survdnn_get_device("auto")
  expect_s3_class(dev, "torch_device")
  # Either cpu or cuda are acceptable
  expect_true(dev$type %in% c("cpu", "cuda"))
})

test_that("survdnn respects .device = 'cpu'", {
  skip_if_not_installed("torch")
  skip_if_not(torch::torch_is_installed())

  veteran <- survival::veteran

  mod_cpu <- survdnn(
    Surv(time, status) ~ age + karno,
    data    = veteran,
    epochs  = 2,
    loss    = "cox",
    verbose = FALSE,
    .device = "cpu"
  )

  expect_equal(mod_cpu$device$type, "cpu")
})
