test_that("cox_loss returns scalar tensor", {
  skip_if_not(torch_is_installed())

  time   <- torch_tensor(runif(20, 1, 100), dtype = torch_float())
  status <- torch_tensor(sample(0:1, 20, replace = TRUE), dtype = torch_float())
  y      <- torch_stack(list(time, status), dim = 2)
  pred   <- torch_randn(20, 1)

  loss <- cox_loss(pred, y)
  expect_true(inherits(loss, "torch_tensor"))
  expect_equal(loss$numel(), 1)
})

test_that("cox_l2_loss returns penalized scalar tensor", {
  skip_if_not(torch_is_installed())

  time   <- torch_tensor(runif(20, 1, 100), dtype = torch_float())
  status <- torch_tensor(sample(0:1, 20, replace = TRUE), dtype = torch_float())
  y      <- torch_stack(list(time, status), dim = 2)
  pred   <- torch_randn(20, 1)

  loss <- cox_l2_loss(pred, y, lambda = 0.01)
  expect_true(inherits(loss, "torch_tensor"))
  expect_equal(loss$numel(), 1)
})

test_that("aft_loss returns scalar tensor (uncensored only)", {
  skip_if_not(torch_is_installed())

  time   <- torch_tensor(runif(20, 1, 100), dtype = torch_float())
  status <- torch_tensor(sample(0:1, 20, replace = TRUE), dtype = torch_float())
  y      <- torch_stack(list(time, status), dim = 2)
  pred   <- torch_randn(20, 1)

  loss <- aft_loss(pred, y)
  expect_true(inherits(loss, "torch_tensor"))
  expect_equal(loss$numel(), 1)
})

test_that("aft_loss returns 0 for fully censored data", {
  skip_if_not(torch_is_installed())

  time   <- torch_tensor(runif(10, 1, 100), dtype = torch_float())
  status <- torch_zeros(10, dtype = torch_float())
  y      <- torch_stack(list(time, status), dim = 2)
  pred   <- torch_randn(10, 1)

  loss <- aft_loss(pred, y)

  expect_equal(as.numeric(loss), 0)
  expect_true(inherits(loss, "torch_tensor"))
  expect_equal(loss$numel(), 1)
})

test_that("aft_loss fully censored returns 0 on same device as pred (if CUDA available)", {
  skip_if_not(torch_is_installed())
  skip_if_not(cuda_is_available())

  device <- torch_device("cuda")

  time   <- torch_tensor(runif(10, 1, 100), dtype = torch_float(), device = device)
  status <- torch_zeros(10, dtype = torch_float(), device = device)
  y      <- torch_stack(list(time, status), dim = 2)
  pred   <- torch_randn(10, 1, device = device)

  loss <- aft_loss(pred, y)

  expect_equal(as.numeric(loss$to(device = "cpu")), 0)
  expect_true(inherits(loss, "torch_tensor"))
  expect_equal(loss$numel(), 1)
  expect_equal(loss$device$type, pred$device$type)
})

test_that("coxtime_loss returns scalar, finite, and non-negative", {
  skip_if_not(torch_is_installed())

  n <- 50
  pred   <- torch_randn(n, 1)
  time   <- torch_rand(n) * 100 + 1e-3
  status <- torch_ones(n)
  true   <- torch_stack(list(time, status), dim = 2)

  loss <- coxtime_loss(pred, true)

  expect_true(inherits(loss, "torch_tensor"))
  expect_equal(loss$numel(), 1)

  # bring to R safely
  loss_num <- as.numeric(loss)
  expect_true(is.finite(loss_num))
  expect_gte(loss_num, 0)
})

test_that("rp_ph_loss returns scalar tensor", {
  skip_if_not(torch_is_installed())

  n <- 40
  pred   <- torch_randn(n, 1)
  time   <- torch_rand(n) * 100 + 1e-3
  status <- torch_tensor(sample(0:1, n, replace = TRUE), dtype = torch_float())
  true   <- torch_stack(list(time, status), dim = 2)

  # minimal knot spec: no internal knots (linear baseline on z)
  knots_internal <- torch_tensor(numeric(0), dtype = torch_float())
  knot_min <- torch_tensor(log(1e-3), dtype = torch_float())
  knot_max <- torch_tensor(log(100), dtype = torch_float())

  gamma <- torch_zeros(2, dtype = torch_float())
  loss <- rp_ph_loss(
    pred = pred, true = true,
    gamma = gamma,
    knots_internal = knots_internal,
    knot_min = knot_min, knot_max = knot_max,
    timescale = "log"
  )

  expect_true(inherits(loss, "torch_tensor"))
  expect_equal(loss$numel(), 1)
  expect_true(is.finite(as.numeric(loss)))
})

test_that("rp_tve_loss returns scalar tensor", {
  skip_if_not(torch_is_installed())

  n <- 40
  pred   <- torch_randn(n, 1)
  time   <- torch_rand(n) * 100 + 1e-3
  status <- torch_tensor(sample(0:1, n, replace = TRUE), dtype = torch_float())
  true   <- torch_stack(list(time, status), dim = 2)

  knots_internal <- torch_tensor(numeric(0), dtype = torch_float())
  knot_min <- torch_tensor(log(1e-3), dtype = torch_float())
  knot_max <- torch_tensor(log(100), dtype = torch_float())

  gamma <- torch_zeros(2, dtype = torch_float())
  alpha <- torch_zeros(2, dtype = torch_float())

  loss <- rp_tve_loss(
    pred = pred, true = true,
    gamma = gamma, alpha = alpha,
    knots_internal = knots_internal,
    knot_min = knot_min, knot_max = knot_max,
    timescale = "log"
  )

  expect_true(inherits(loss, "torch_tensor"))
  expect_equal(loss$numel(), 1)
  expect_true(is.finite(as.numeric(loss)))
})

test_that("survdnn_make_loss returns fn and params for RP losses", {
  skip_if_not(torch_is_installed())

  n <- 40
  time   <- torch_rand(n) * 100 + 1e-3
  status <- torch_tensor(sample(0:1, n, replace = TRUE), dtype = torch_float())
  y      <- torch_stack(list(time, status), dim = 2)

  device <- torch_device("cpu")

  obj1 <- survdnn_make_loss("rp_ph", y_tensor = y, device = device)
  expect_true(is.list(obj1))
  expect_true(is.function(obj1$fn))
  expect_true(is.list(obj1$params))
  expect_gte(length(obj1$params), 1)

  obj2 <- survdnn_make_loss("rp_tve", y_tensor = y, device = device)
  expect_true(is.list(obj2))
  expect_true(is.function(obj2$fn))
  expect_true(is.list(obj2$params))
  expect_gte(length(obj2$params), 2)
})
