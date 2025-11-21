#' Loss Functions for survdnn Models
#'
#' These functions define various loss functions used internally by `survdnn()` for training deep neural networks on right-censored survival data.
#'
#' @section Supported Losses:
#' - **Cox partial likelihood loss** (`cox_loss`): Negative partial log-likelihood used in proportional hazards modeling.
#' - **L2-penalized Cox loss** (`cox_l2_loss`): Adds L2 regularization to the Cox loss.
#' - **Accelerated Failure Time (AFT) loss** (`aft_loss`): Mean squared error between predicted and log-transformed event times, applied to uncensored observations only.
#' - **CoxTime loss** (`coxtime_loss`): Implements the partial likelihood loss from Kvamme & Borgan (2019), used in Cox-Time models.
#'
#' @param pred A tensor of predicted values (typically linear predictors or log-times).
#' @param true A tensor with two columns: observed time and status (1 = event, 0 = censored).
#' @param lambda Regularization parameter for `cox_l2_loss` (default: `1e-4`).
#'
#' @return A scalar `torch_tensor` representing the loss value.
#' @name survdnn_losses
#' @keywords internal
#' @examples
#' # Used internally by survdnn()
NULL


#' @rdname survdnn_losses
#' @export
cox_loss <- function(pred, true) {
  time <- true[, 1]
  status <- true[, 2]

  idx <- torch_argsort(time, descending = TRUE)
  time <- time[idx]
  status <- status[idx]
  pred <- -pred[idx, 1]  # negate for log-partial likelihood

  log_cumsum_exp <- torch_logcumsumexp(pred, dim = 1)
  event_mask <- (status == 1)

  loss <- -torch_mean(pred[event_mask] - log_cumsum_exp[event_mask])
  loss
}


#' @rdname survdnn_losses
#' @export
cox_l2_loss <- function(pred, true, lambda = 1e-4) {
  base_loss <- cox_loss(pred, true)
  l2_penalty <- lambda * torch_mean(pred^2)
  base_loss + l2_penalty
}


#' @rdname survdnn_losses
#' @export
aft_loss <- function(pred, true) {
  time <- true[, 1]
  status <- true[, 2]
  log_time <- torch_log(time)

  event_mask <- (status == 1)
  n_events <- as.numeric(torch_sum(event_mask))

  if (n_events == 0) {
    return(torch_zeros_like(pred[1, 1])) ## this ensure the returned loss has the same device as pred & has the same dtype as pred (CPU/CUDA/MPS)
  }

  pred_event <- pred[event_mask, 1]
  log_time_event <- log_time[event_mask]

  torch_mean((pred_event - log_time_event)^2)
}


#' @rdname survdnn_losses
#' @export
coxtime_loss <- function(pred, true) {

  # `pred` is a tensor of shape [n, 1]: g(t_i, x_i)
  # `true` is a tensor with columns: time and status

  time <- true[, 1]
  status <- true[, 2]
  n <- time$size()[[1]]

  # sorting by time descending
  idx <- torch_argsort(time, descending = TRUE)
  time <- time[idx]
  status <- status[idx]
  pred <- pred[idx, 1]  # ensure shape [n]

  # compute risk set matrix: R_ij = 1 if time_j >= time_i
  time_i <- time$view(c(n, 1))           # [n, 1]
  time_j <- time$view(c(1, n))           # [1, n]
  risk_matrix <- (time_j >= time_i)$to(dtype = torch_float())  # [n, n]

  # compute difference: g(t_i, x_j) - g(t_i, x_i)
  pred_i <- pred$view(c(n, 1))           # [n, 1]
  pred_j <- pred$view(c(1, n))           # [1, n]
  diff <- pred_j - pred_i                # [n, n]

  # mask for events only
  event_mask <- (status == 1)

  # compute log sum exp over risk set
  log_sum_exp <- torch_logsumexp(diff * risk_matrix, dim = 2)  # [n]

  # final partial likelihood loss: mean over events only
  loss_terms <- log_sum_exp[event_mask]
  loss <- torch_mean(loss_terms)
  return(loss)
}
