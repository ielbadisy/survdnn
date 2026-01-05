test_that("cox_loss returns scalar tensor (and is differentiable)", {
  skip_if_not(torch_is_installed())

  time   <- torch_tensor(runif(20, 1, 100))
  status <- torch_tensor(sample(0:1, 20, replace = TRUE))
  y      <- torch_stack(list(time, status), dim = 2)

  # pred must require grad to test autograd
  pred <- torch_randn(20, 1, requires_grad = TRUE)

  loss <- cox_loss(pred, y)

  expect_true(inherits(loss, "torch_tensor"))
  expect_equal(loss$numel(), 1)
  expect_true(isTRUE(loss$requires_grad))

  expect_silent(loss$backward())
  expect_false(torch::is_undefined_tensor(pred$grad))
})

test_that("cox_l2_loss returns penalized scalar tensor (and is differentiable)", {
  skip_if_not(torch_is_installed())

  time   <- torch_tensor(runif(20, 1, 100))
  status <- torch_tensor(sample(0:1, 20, replace = TRUE))
  y      <- torch_stack(list(time, status), dim = 2)

  pred <- torch_randn(20, 1, requires_grad = TRUE)

  loss <- cox_l2_loss(pred, y, lambda = 0.01)

  expect_true(inherits(loss, "torch_tensor"))
  expect_equal(loss$numel(), 1)
  expect_true(isTRUE(loss$requires_grad))

  expect_silent(loss$backward())
  expect_false(torch::is_undefined_tensor(pred$grad))
})

test_that("aft_loss returns scalar tensor (censored log-normal NLL)", {
  skip_if_not(torch_is_installed())

  time   <- torch_tensor(runif(20, 1, 100))
  status <- torch_tensor(sample(0:1, 20, replace = TRUE))
  y      <- torch_stack(list(time, status), dim = 2)

  pred <- torch_randn(20, 1, requires_grad = TRUE)

  loss <- aft_loss(pred, y, sigma = 1)

  expect_true(inherits(loss, "torch_tensor"))
  expect_equal(loss$numel(), 1)
  expect_true(isTRUE(loss$requires_grad))

  expect_silent(loss$backward())
  expect_false(torch::is_undefined_tensor(pred$grad))
})

test_that("aft_loss is finite for fully censored data (no special-casing to 0)", {
  skip_if_not(torch_is_installed())

  time   <- torch_tensor(runif(10, 1, 100))
  status <- torch_zeros(10)
  y      <- torch_stack(list(time, status), dim = 2)

  pred <- torch_randn(10, 1, requires_grad = TRUE)

  loss <- aft_loss(pred, y, sigma = 1)

  expect_true(inherits(loss, "torch_tensor"))
  expect_equal(loss$numel(), 1)
  expect_true(is.finite(as.numeric(loss)))
  expect_gt(as.numeric(loss), 0)

  expect_silent(loss$backward())
  expect_false(torch::is_undefined_tensor(pred$grad))
})

test_that("aft_loss fully censored stays on same device as pred (CUDA)", {
  skip_if_not(torch_is_installed())
  skip_if_not(cuda_is_available())

  device <- torch_device("cuda")

  time   <- torch_tensor(runif(10, 1, 100), device = device)
  status <- torch_zeros(10, device = device)
  y      <- torch_stack(list(time, status), dim = 2)

  pred <- torch_randn(10, 1, device = device, requires_grad = TRUE)

  loss <- aft_loss(pred, y, sigma = 1)

  expect_true(inherits(loss, "torch_tensor"))
  expect_equal(loss$numel(), 1)
  expect_true(is.finite(as.numeric(loss$to(device = "cpu"))))
  expect_equal(loss$device$type, pred$device$type)

  expect_silent(loss$backward())
  expect_false(torch::is_undefined_tensor(pred$grad))
})

test_that("coxtime_loss() is a placeholder and errors (use factory in training)", {
  skip_if_not(torch_is_installed())

  n <- 50
  pred   <- torch_randn(c(n, 1))
  time   <- torch_rand(c(n, 1)) * 100
  status <- torch_ones(c(n, 1))
  true   <- torch_cat(list(time, status), dim = 2)

  expect_error(
    coxtime_loss(pred, true),
    regexp = "not identifiable|requires evaluating",
    fixed  = FALSE
  )
})

test_that("survdnn__coxtime_loss_factory returns scalar tensor and is differentiable", {
  skip_if_not(torch_is_installed())

  n <- 30
  p <- 4
  d <- 1 + p

  # simple net; no BN to keep this unit test lightweight
  net <- torch::nn_sequential(
    torch::nn_linear(d, 8),
    torch::nn_relu(),
    torch::nn_linear(8, 1)
  )
  net$train()

  time   <- torch_tensor(runif(n, 1, 100))
  status <- torch_tensor(sample(0:1, n, replace = TRUE))
  y      <- torch_stack(list(time, status), dim = 2)

  x_cov  <- torch_randn(n, p)
  x_tensor <- torch_cat(list(time$unsqueeze(2), x_cov), dim = 2) # [n, 1+p]

  lf <- survdnn__coxtime_loss_factory(net)
  loss <- lf(x_tensor, y)

  expect_true(inherits(loss, "torch_tensor"))
  expect_equal(loss$numel(), 1)
  expect_true(isTRUE(loss$requires_grad))

  expect_silent(loss$backward())

  # at least one parameter should have a defined gradient
  grads <- lapply(net$parameters, function(par) par$grad)
  gsum <- sum(vapply(grads, function(g) if (is.null(g)) 0 else as.numeric(g$abs()$sum()$item()), numeric(1)))
  expect_gt(gsum, 0)
})