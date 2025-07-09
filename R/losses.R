
cox_loss <- function(pred, true) {
  time <- true[, 1]
  status <- true[, 2]

  idx <- torch_argsort(time, descending = TRUE)
  time <- time[idx]
  status <- status[idx]
  pred <- -pred[idx, 1]  # flip sign to convert to negative risk

  log_cumsum_exp <- torch_logcumsumexp(pred, dim = 1)
  event_mask <- (status == 1)

  loss <- -torch_mean(pred[event_mask] - log_cumsum_exp[event_mask])
  return(loss)
}




cox_l2_loss <- function(pred, true, lambda = 1e-4) {
  loss <- cox_loss(pred, true)
  l2_penalty <- lambda * torch_mean(pred^2)
  return(loss + l2_penalty)
}




rank_loss <- function(pred, true) {
  time <- true[, 1]
  status <- true[, 2]
  n <- pred$size(1)

  loss_terms <- list()
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      ti <- as.numeric(time[i])
      tj <- as.numeric(time[j])
      si <- as.numeric(status[i])
      sj <- as.numeric(status[j])

      # comparable pairs: one event, different times
      if ((si == 1 && ti < tj) || (sj == 1 && tj < ti)) {
        d_ij <- pred[i, 1] - pred[j, 1]
        y_ij <- if (ti < tj) 1 else -1

        hinge <- torch_relu(1 - y_ij * d_ij)
        loss_terms <- append(loss_terms, list(hinge))
      }
    }
  }

  if (length(loss_terms) == 0) return(torch_tensor(0.0, dtype = torch_float()))
  loss_tensor <- torch_stack(loss_terms)
  return(torch_mean(loss_tensor))
}



aft_loss <- function(pred, true) {
  time <- true[, 1]
  status <- true[, 2]
  log_time <- torch_log(time)

  event_mask <- (status == 1)
  n_events <- as.numeric(torch_sum(event_mask))

  if (n_events == 0) {
    return(torch_tensor(0, dtype = torch_float()))
  }

  pred_event <- pred[event_mask, 1]
  log_time_event <- log_time[event_mask]

  mse <- torch_mean((pred_event - log_time_event)^2)
  return(mse)
}






# internal validator for .loss_fn (not exported)
validate_loss_fn <- function(.loss_fn) {
  test_pred <- torch_randn(10, 1)
  test_true <- torch_cat(list(
    torch_rand(10, 1) * 100,          # random times
    torch_randint(low = 0, high = 2, size = c(10, 1)) # event indicator
  ), dim = 2)

  test_loss <- try(.loss_fn(test_pred, test_true), silent = TRUE)

  if (inherits(test_loss, "try-error") || !inherits(test_loss, "torch_tensor") ||
      test_loss$numel() != 1) {
    stop(".loss_fn must return a scalar torch tensor with shape (1)")
  }
}




