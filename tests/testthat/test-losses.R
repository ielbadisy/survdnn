library(testthat)
library(torch)

test_that("cox_loss returns scalar tensor", {
  time <- torch_tensor(runif(20, 1, 100))
  status <- torch_tensor(sample(0:1, 20, replace = TRUE))
  y <- torch_stack(list(time, status), dim = 2)
  pred <- torch_randn(20, 1)

  loss <- cox_loss(pred, y)
  expect_true(inherits(loss, "torch_tensor"))
  expect_equal(loss$numel(), 1)
})

test_that("cox_l2_loss returns penalized scalar tensor", {
  time <- torch_tensor(runif(20, 1, 100))
  status <- torch_tensor(sample(0:1, 20, replace = TRUE))
  y <- torch_stack(list(time, status), dim = 2)
  pred <- torch_randn(20, 1)

  loss <- cox_l2_loss(pred, y, lambda = 0.01)
  expect_true(inherits(loss, "torch_tensor"))
  expect_equal(loss$numel(), 1)
})

test_that("rank_loss returns scalar tensor", {
  time <- torch_tensor(runif(10, 1, 100))
  status <- torch_tensor(sample(0:1, 10, replace = TRUE))
  y <- torch_stack(list(time, status), dim = 2)
  pred <- torch_randn(10, 1)

  loss <- rank_loss(pred, y)
  expect_true(inherits(loss, "torch_tensor"))
  expect_equal(loss$numel(), 1)
})

test_that("aft_loss returns scalar tensor (uncensored only)", {
  time <- torch_tensor(runif(20, 1, 100))
  status <- torch_tensor(sample(0:1, 20, replace = TRUE))
  y <- torch_stack(list(time, status), dim = 2)
  pred <- torch_randn(20, 1)

  loss <- aft_loss(pred, y)
  expect_true(inherits(loss, "torch_tensor"))
  expect_equal(loss$numel(), 1)
})

test_that("aft_loss returns 0 for fully censored data", {
  time <- torch_tensor(runif(10, 1, 100))
  status <- torch_tensor(torch_zeros(10))
  y <- torch_stack(list(time, status), dim = 2)
  pred <- torch_randn(10, 1)

  loss <- aft_loss(pred, y)
  expect_equal(as.numeric(loss), 0)
})
