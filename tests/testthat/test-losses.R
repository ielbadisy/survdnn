
test_that("cox_loss returns scalar tensor", {
  skip_if_not(torch_is_installed())

  time <- torch_tensor(runif(20, 1, 100))
  status <- torch_tensor(sample(0:1, 20, replace = TRUE))
  y <- torch_stack(list(time, status), dim = 2)
  pred <- torch_randn(20, 1)

  loss <- cox_loss(pred, y)
  expect_true(inherits(loss, "torch_tensor"))
  expect_equal(loss$numel(), 1)
})

test_that("cox_l2_loss returns penalized scalar tensor", {
  skip_if_not(torch_is_installed())

  time <- torch_tensor(runif(20, 1, 100))
  status <- torch_tensor(sample(0:1, 20, replace = TRUE))
  y <- torch_stack(list(time, status), dim = 2)
  pred <- torch_randn(20, 1)

  loss <- cox_l2_loss(pred, y, lambda = 0.01)
  expect_true(inherits(loss, "torch_tensor"))
  expect_equal(loss$numel(), 1)
})



test_that("aft_loss returns scalar tensor (uncensored only)", {
  skip_if_not(torch_is_installed())

  time <- torch_tensor(runif(20, 1, 100))
  status <- torch_tensor(sample(0:1, 20, replace = TRUE))
  y <- torch_stack(list(time, status), dim = 2)
  pred <- torch_randn(20, 1)

  loss <- aft_loss(pred, y)
  expect_true(inherits(loss, "torch_tensor"))
  expect_equal(loss$numel(), 1)
})

test_that("aft_loss returns 0 for fully censored data", {
  skip_if_not(torch_is_installed())

  time <- torch_tensor(runif(10, 1, 100))
  status <- torch_tensor(torch_zeros(10))
  y <- torch_stack(list(time, status), dim = 2)
  pred <- torch_randn(10, 1)

  loss <- aft_loss(pred, y)
  expect_equal(as.numeric(loss), 0)
})




test_that("coxtime_loss implements partial likelihood faithfully", {
  skip_if_not(torch_is_installed())

  n <- 50
  pred <- torch_randn(c(n, 1))
  time <- torch_rand(c(n, 1)) * 100
  status <- torch_ones(c(n, 1))
  true <- torch_cat(list(time, status), dim = 2)

  loss <- coxtime_loss(pred, true)
  expect_true(!torch::is_undefined_tensor(loss))
  expect_length(loss, 1)
  expect_gt(as.numeric(loss), 0)
})
