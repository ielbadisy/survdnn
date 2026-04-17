test_that("cv_survdnn performs cross-validation and returns fold-level metrics", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran

  out <- cv_survdnn(
    Surv(time, status) ~ age + karno + celltype,
    data = data,
    times = c(30, 90),
    metrics = c("cindex", "ibs"),
    folds = 2,
    .seed = 123,
    .threads = 1,
    hidden = c(8),
    epochs = 5,
    verbose = FALSE
  )

  expect_s3_class(out, "data.frame")
  expect_true(all(c("fold", "metric", "value") %in% names(out)))
  expect_true(all(out$fold %in% 1:2))
})

test_that("cv_survdnn errors on missing inputs or bad arguments", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  df <- survival::veteran

  expect_error(
    cv_survdnn("not a formula", df, times = 50),
    "`formula` must be a survival formula"
  )

  expect_error(
    cv_survdnn(Surv(time, status) ~ ., "not a df", times = 50),
    "`data` must be a data frame"
  )

  expect_error(
    cv_survdnn(Surv(time, status) ~ ., df),
    "must provide a `times`"
  )
})

test_that("cv_survdnn verbose controls progress messages", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran

  msg_on <- capture.output(
    cv_survdnn(
      Surv(time, status) ~ age + karno + celltype,
      data = data,
      times = c(30),
      metrics = c("ibs"),
      folds = 2,
      .seed = 123,
      .threads = 1,
      hidden = c(8),
      epochs = 1,
      verbose = TRUE
    ),
    type = "message"
  )

  expect_true(any(grepl("\\[survdnn::cv\\]", msg_on)))
  expect_true(any(grepl("\\[survdnn::fit\\]", msg_on)))

  msg_off <- capture.output(
    cv_survdnn(
      Surv(time, status) ~ age + karno + celltype,
      data = data,
      times = c(30),
      metrics = c("ibs"),
      folds = 2,
      .seed = 123,
      .threads = 1,
      hidden = c(8),
      epochs = 1,
      verbose = FALSE
    ),
    type = "message"
  )

  expect_false(any(grepl("\\[survdnn::cv\\]|\\[survdnn::fit\\]|\\[survdnn::eval\\]", msg_off)))
})
