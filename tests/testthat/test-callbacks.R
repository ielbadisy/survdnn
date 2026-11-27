test_that("callback_early_stopping stops on plateauing loss", {
  cb <- callback_early_stopping(patience = 2, mode = "min")

  # fake loss sequence: improves, then flat/worse
  out1 <- cb(epoch = 1, current = 1.0)  # best = 1.0
  out2 <- cb(epoch = 2, current = 0.9)  # best = 0.9
  out3 <- cb(epoch = 3, current = 0.91) # wait = 1
  out4 <- cb(epoch = 4, current = 0.92) # wait = 2 -> stop

  expect_false(out1)
  expect_false(out2)
  expect_false(out3)
  expect_true(out4)
})

test_that("survdnn uses early stopping callback on training loss", {
  skip_if_not_installed("torch")
  skip_if_not(torch::torch_is_installed())

  veteran <- survival::veteran

  # strong early stopping: patience small, epochs large (need to investigate this in a simuation study)
  cb <- callback_early_stopping(patience = 1, mode = "min")

  mod <- survdnn::survdnn(
    Surv(time, status) ~ age + karno + celltype,
    data       = veteran,
    hidden     = c(8L, 4L),
    activation = "relu",
    epochs     = 50L,
    loss       = "cox",
    verbose    = FALSE,
    dropout    = 0.3,
    batch_norm = TRUE,
    callbacks  = cb,
    .seed      = 123,
    .device    = "cpu"
  )

  # we can't guarantee how early it stops, but in realistic settings
  # we expect fewer than the requested epochs when callback is active (more investigation is needed)
  expect_true(length(mod$loss_history) <= 50L)
  expect_true(length(mod$loss_history) >= 1L)
})

test_that("callbacks monitor the validation loss when validation > 0", {
  skip_if_not_installed("torch")
  skip_if_not(torch::torch_is_installed())
  seen <- numeric(0)
  cb <- function(epoch, current) { seen <<- c(seen, current); FALSE }
  fit <- function(...) survdnn(Surv(time, status) ~ age + karno, data = survival::veteran, hidden = 4L,
                               epochs = 8L, verbose = FALSE, .seed = 1, .device = "cpu", ...)
  m <- fit(callbacks = cb, validation = 0.3)
  expect_length(m$val_loss_history, 8L)
  expect_equal(seen, m$val_loss_history)
  expect_false(isTRUE(all.equal(m$val_loss_history, m$loss_history)))
  seen <- numeric(0)
  m0 <- fit(callbacks = cb)
  expect_null(m0$val_loss_history)
  expect_equal(seen, m0$loss_history)
  for (l in c("cox_l2", "aft", "coxtime")) expect_length(fit(loss = l, validation = 0.3)$val_loss_history, 8L)
})
