#' Loss Functions for survdnn Models
#'
#' These functions define various loss functions used to train `survdnn` deep learning survival models.
#'
#' @param pred A tensor of predicted values from the neural network (typically of shape `[n, 1]`).
#' @param true A tensor with two columns: observed time and event indicator.
#' @param lambda Regularization strength (for `cox_l2_loss`).
#'
#' @return A scalar torch tensor representing the loss.
#' @rdname survdnn_losses
#' @export
#' @examples
#' library(torch)
#' library(survival)
#'
#' # Simulated survival data
#' set.seed(123)
#' n <- 100
#' toy_data <- data.frame(
#'   x1 = rnorm(n),
#'   x2 = rbinom(n, 1, 0.5),
#'   time = rexp(n, 0.1),
#'   status = rbinom(n, 1, 0.7)
#' )
#'
#' # Cox loss
#' mod_cox <- survdnn(Surv(time, status) ~ x1 + x2,
#'                    data = toy_data,
#'                    .loss_fn = cox_loss,
#'                    epochs = 100,
#'                    verbose = TRUE)
#' plot(mod_cox$loss_history, type = "l", main = "cox_loss")
#'
#' # Cox + L2 penalty
#' mod_cox_l2 <- survdnn(Surv(time, status) ~ x1 + x2,
#'                    data = toy_data,
#'                    .loss_fn = cox_l2_loss,
#'                    epochs = 100,
#'                    verbose = TRUE)
#' plot(mod_cox_l2$loss_history, type = "l", main = "cox_l2_loss")
#'
#' # AFT loss
#' mod_aft <- survdnn(Surv(time, status) ~ x1 + x2,
#'                    data = toy_data,
#'                    .loss_fn = aft_loss,
#'                    epochs = 100,
#'                    verbose = TRUE)
#' plot(mod_aft$loss_history, type = "l", main = "aft_loss")
#'
#' # Custom loss
#' combo_loss <- function(pred, true) {
#'   time <- true[, 1]
#'   torch_mean((pred - log(time + 1))^2) + 0.01 * torch_mean(pred)
#' }
#'
#' mod_combo <- survdnn(Surv(time, status) ~ x1 + x2,
#'                    data = toy_data,
#'                    .loss_fn = combo_loss,
#'                    epochs = 100,
#'                    verbose = TRUE)
#' plot(mod_combo$loss_history, type = "l", main = "combo_loss")
cox_loss <- function(pred, true) {
  time <- true[, 1]
  status <- true[, 2]

  idx <- torch_argsort(time, descending = TRUE)
  time <- time[idx]
  status <- status[idx]
  pred <- -pred[idx, 1]

  log_cumsum_exp <- torch_logcumsumexp(pred, dim = 1)
  event_mask <- (status == 1)

  loss <- -torch_mean(pred[event_mask] - log_cumsum_exp[event_mask])
  return(loss)
}

#' @rdname survdnn_losses
#' @export
cox_l2_loss <- function(pred, true, lambda = 1e-4) {
  loss <- cox_loss(pred, true)
  l2_penalty <- lambda * torch_mean(pred^2)
  return(loss + l2_penalty)
}

#' @rdname survdnn_losses
#' @export
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

#' @rdname survdnn_losses
#' @export
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

#' @rdname survdnn_losses
#' @export
combo_loss <- function(pred, true) {
  time <- true[, 1]
  torch_mean((pred - log(time + 1))^2) + 0.01 * torch_mean(pred)
}

#' Validate a Custom Loss Function
#'
#' Ensures that a user-supplied loss function is compatible with `survdnn` training requirements.
#'
#' @param .loss_fn A function that takes two arguments (`pred`, `true`) and returns a scalar torch tensor.
#'
#' @return NULL. Throws an error if validation fails.
#' @export
#' @examples
#' validate_loss_fn(cox_loss)
#' validate_loss_fn(aft_loss)
#' validate_loss_fn(function(pred, true) torch_mean((pred - true[, 1])^2))
validate_loss_fn <- function(.loss_fn) {
  test_pred <- torch_randn(10, 1)
  test_true <- torch_cat(list(
    torch_rand(10, 1) * 100,
    torch_randint(low = 0, high = 2, size = c(10, 1))
  ), dim = 2)

  test_loss <- try(.loss_fn(test_pred, test_true), silent = TRUE)

  if (inherits(test_loss, "try-error") || !inherits(test_loss, "torch_tensor") ||
      test_loss$numel() != 1) {
    stop(".loss_fn must return a scalar torch tensor with shape (1)")
  }
}
