
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
