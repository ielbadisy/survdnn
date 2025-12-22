test_that("survdnn omits NAs and messages when verbose=TRUE", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())


  dat <- data.frame(
    time  = c(1, 2, 3, 4),
    status = c(1, 1, 0, 1),
    x1    = c(0.1, NA, 0.3, 0.4)
  )

  expect_message(
    survdnn(
      survival::Surv(time, status) ~ x1,
      data = dat,
      na_action = "omit",
      verbose = TRUE,
      epochs = 1
    ),
    "Removed 1 observations with missing values"
  )
})

test_that("survdnn fails on NAs when na_action='fail'", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())


  dat <- data.frame(
    time  = c(1, 2, 3, 4),
    status = c(1, 1, 0, 1),
    x1    = c(0.1, NA, 0.3, 0.4)
  )

  expect_error(
    survdnn(
      survival::Surv(time, status) ~ x1,
      data = dat,
      na_action = "fail",
      verbose = FALSE,
      epochs = 1
    )
  )
})

test_that("evaluate_survdnn respects na_action='fail' on newdata", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())


  train <- data.frame(
    time  = c(1, 2, 3, 4, 5, 6),
    status = c(1, 1, 0, 1, 0, 1),
    x1    = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
  )

  mod <- survdnn(
    survival::Surv(time, status) ~ x1,
    data = train,
    epochs = 2,
    verbose = FALSE,
    na_action = "fail"
  )

  newdat <- data.frame(
    time  = c(1, 2, 3),
    status = c(1, 0, 1),
    x1    = c(0.1, NA, 0.3)
  )

  expect_error(
    evaluate_survdnn(
      mod,
      metrics = "cindex",
      times = 2,
      newdata = newdat,
      na_action = "fail"
    )
  )
})



test_that("evaluate_survdnn omits incomplete rows when na_action='omit' (no crash)", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())


  train <- data.frame(
    time   = c(1, 2, 3, 4, 5, 6),
    status = c(1, 1, 0, 1, 0, 1),
    x1     = c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
  )

  mod <- survdnn(
    survival::Surv(time, status) ~ x1,
    data = train,
    epochs = 2,
    verbose = FALSE,
    na_action = "fail"
  )

  newdat <- data.frame(
    time   = c(1, 2, 3),
    status = c(1, 0, 1),
    x1     = c(0.1, NA, 0.3)
  )

  res <- suppressWarnings(
    evaluate_survdnn(
      mod,
      metrics = c("brier", "ibs"),
      times = c(1, 2),
      newdata = newdat,
      na_action = "omit",
      verbose = TRUE
    )
  )

  expect_s3_class(res, "tbl_df")
  expect_true(all(c("metric", "value") %in% names(res)))
})
