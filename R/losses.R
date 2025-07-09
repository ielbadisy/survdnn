
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
