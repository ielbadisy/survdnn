#' Loss Functions for survdnn Models
#'
#' These functions define various loss functions used internally by `survdnn()`
#' for training deep neural networks on right-censored survival data.
#'
#' @section Supported Losses:
#' - **Cox partial likelihood loss** (`cox_loss`): Negative partial log-likelihood used in proportional hazards modeling.
#' - **L2-penalized Cox loss** (`cox_l2_loss`): Adds L2 regularization to the Cox loss.
#' - **Accelerated Failure Time (AFT) loss** (`aft_loss`): Log-normal AFT **censored negative log-likelihood**
#'   (uses both events and censored observations).
#' - **CoxTime loss** (`coxtime_loss`): Placeholder (see details). A correct CoxTime loss requires access to the network and the full input tensor.
#'
#' @param pred A torch tensor of model predictions. Its interpretation depends
#'   on the loss function:
#'   \itemize{
#'     \item{\code{loss = "cox"} or \code{"cox_l2"}: linear predictors
#'           (log hazard ratios).}
#'     \item{\code{loss = "aft"}: predicted log survival times.}
#'     \item{\code{loss = "coxtime"}: predicted time-dependent risk scores.}
#'   }
#' @param true A tensor with two columns: observed time and status (1 = event, 0 = censored).
#' @param lambda Regularization parameter for `cox_l2_loss` (default: `1e-3`).
#' @param sigma Positive numeric scale parameter for the log-normal AFT model (default: `1`).
#'   In `survdnn()`, a learnable global scale can be used via `survdnn__aft_lognormal_nll_factory()`.
#' @param aft_loc Numeric scalar location offset for the AFT model on the log-time scale.
#'   When non-zero, the model is trained on centered log-times `log(time) - aft_loc` for better numerical stability.
#'   Prediction should add this offset back: `mu = mu_resid + aft_loc`.
#' @param eps Small constant for numerical stability (default: `1e-12`).
#'
#' @return A scalar `torch_tensor` representing the loss value.
#' @name survdnn_losses
#' @keywords internal
#' @examples
#' # Used internally by survdnn()
NULL


# internal utilities
#' @keywords internal
survdnn__zeros_like_scalar <- function(x) {
  torch::torch_zeros_like(x$view(c(1)))[1]
}

#' @keywords internal
survdnn__count_true <- function(mask) {
  as.integer(mask$sum()$item())
}

#' @keywords internal
survdnn__log_surv_std_normal <- function(z, eps = 1e-12) {
  sqrt2 <- torch::torch_sqrt(torch::torch_tensor(2, dtype = z$dtype, device = z$device))
  u     <- z / sqrt2
  S     <- torch::torch_clamp(0.5 * (1 - torch::torch_erf(u)), min = eps)
  torch::torch_log(S)
}


# Cox loss 

#' @rdname survdnn_losses
#' @export
cox_loss <- function(pred, true) {
  time   <- true[, 1]
  status <- true[, 2]

  idx <- torch::torch_argsort(time, descending = TRUE)
  status <- status[idx]

  lp <- -pred[idx, 1]

  event_mask <- (status == 1)
  if (survdnn__count_true(event_mask) == 0) {
    return(survdnn__zeros_like_scalar(lp[1]))
  }

  log_cumsum_exp <- torch::torch_logcumsumexp(lp, dim = 1)
  -torch::torch_mean(lp[event_mask] - log_cumsum_exp[event_mask])
}

#' @rdname survdnn_losses
#' @export
cox_l2_loss <- function(pred, true, lambda = 1e-3) {
  base_loss <- cox_loss(pred, true)
  lp <- -pred[, 1]
  l2_penalty <- lambda * torch::torch_mean(lp^2)
  base_loss + l2_penalty
}


# AFT loss (log-normal AFT censored negative log-likelihood)

#' @rdname survdnn_losses
#' @export
aft_loss <- function(pred, true, sigma = 1, aft_loc = 0, eps = 1e-12) {

  time   <- true[, 1]
  status <- true[, 2]

  t  <- torch::torch_clamp(time, min = eps)
  lt <- torch::torch_log(t)

  mu_resid <- pred[, 1]

  sigma_t <- torch::torch_tensor(
    as.numeric(sigma),
    dtype  = mu_resid$dtype,
    device = mu_resid$device
  )

  sigma_t <- torch::torch_clamp(sigma_t, min = eps)
  log_sigma <- torch::torch_log(sigma_t)

  aft_loc_t <- torch::torch_tensor(
    as.numeric(aft_loc),
    dtype  = mu_resid$dtype,
    device = mu_resid$device
  )
  
  lt_c <- lt - aft_loc_t

  # In the log-normal AFT model, log(T) = aft_loc + mu + sigma * Z,
  # with Z ~ N(0, 1). Here, `pred` represents the subject-specific
  # location term mu, while `sigma` controls global time dispersion.

  z <- (lt_c - mu_resid) / sigma_t
  
  logS <- survdnn__log_surv_std_normal(z, eps = eps)

  nll_event <- lt + log_sigma + 0.5 * z^2
  nll_cens  <- -logS

  nll <- torch::torch_where(status == 1, nll_event, nll_cens)
  torch::torch_mean(nll)
}



# CoxTime loss (not identifiable from pred alone)

#' @rdname survdnn_losses
#' @export
coxtime_loss <- function(pred, true) {
  stop(
    "coxtime_loss(pred, true) is not identifiable from `pred` alone.\n",
    "Cox-Time requires evaluating g(t_i, x_j) for all subjects j at each event time t_i.\n",
    "Use the internal factory `survdnn__coxtime_loss_factory()` from survdnn() where `net` and the full input tensor are available.",
    call. = FALSE
  )
}


# Internals (with  loss factory pattern)


# Correct CoxTime
# IMPORTANT FIX:
# - use `true[,1]` (RAW time) for sorting + risk sets
# - use `x_tensor[,1]` (TIME AS FED TO NET) when calling net

#' @keywords internal
survdnn__coxtime_loss_factory <- function(net) {

  force(net)

  function(x_tensor, true, eps = 1e-12) {

    time_raw <- true[, 1]
    status   <- true[, 2]
    n        <- time_raw$size()[[1]]

    d <- x_tensor$size()[[2]]
    if (d < 2) stop("CoxTime expects x_tensor with at least 2 columns: (time, x).", call. = FALSE)

    time_inp <- x_tensor[, 1]               # time as used by the net
    x_cov    <- x_tensor[, 2:d, drop = FALSE]

    ## sort by RAW time (risk sets depend on raw ordering. Not so precise but approximate the risk intensity)
    idx <- torch::torch_argsort(time_raw, descending = TRUE)

    time_raw <- time_raw[idx]
    time_inp <- time_inp[idx]
    status   <- status[idx]
    x_cov    <- x_cov[idx, , drop = FALSE]

    event_mask <- (status == 1)
    ne <- as.integer(event_mask$sum()$item())
    if (ne == 0) return(torch::torch_zeros_like(time_raw[1]))

    ## event times
    t_event_raw <- time_raw[event_mask]     # for risk sets
    t_event_inp <- time_inp[event_mask]     # for net input
    x_event     <- x_cov[event_mask, , drop = FALSE]

    ## numerator: g(t_i, x_i) for events
    inp_num <- torch::torch_cat(list(t_event_inp$unsqueeze(2), x_event), dim = 2)
    g_num   <- net(inp_num)[, 1]

    ## denominator: for each event time t_i, evaluate g(t_i, x_j) for all j
    p <- x_cov$size()[[2]]

    x_rep <- x_cov$unsqueeze(2)$expand(c(n, ne, p))$permute(c(2, 1, 3))
    t_rep <- t_event_inp$view(c(ne, 1, 1))$expand(c(ne, n, 1))  # time for net input
    inp_den <- torch::torch_cat(list(t_rep, x_rep), dim = 3)

    inp_den2 <- inp_den$reshape(c(ne * n, d))
    g_den2   <- net(inp_den2)[, 1]
    g_den    <- g_den2$reshape(c(ne, n))

    ## risk sets computed on RAW time
    time_j <- time_raw$view(c(1, n))
    t_i    <- t_event_raw$view(c(ne, 1))
    risk   <- (time_j >= t_i)

    neg_inf  <- torch::torch_tensor(-Inf, dtype = g_den$dtype, device = g_den$device)
    g_masked <- torch::torch_where(risk, g_den, neg_inf)

    log_denom <- torch::torch_logsumexp(g_masked, dim = 2)
    -torch::torch_mean(g_num - log_denom)
  }
}


# AFT log-normal censored NLL factory (learnable global log(sigma))
# with optional centering by aft_loc.

#' @keywords internal
survdnn__aft_lognormal_nll_factory <- function(device, aft_loc = 0) {

  log_sigma <- torch::torch_tensor(
    0,
    dtype = torch::torch_float(),
    device = device,
    requires_grad = TRUE
  )

  aft_loc_t <- torch::torch_tensor(
    as.numeric(aft_loc),
    dtype = torch::torch_float(),
    device = device
  )

  loss_fn <- function(net, x, y, eps = 1e-12) {

    time   <- y[, 1]
    status <- y[, 2]

    mu_resid <- net(x)[, 1]

    t  <- torch::torch_clamp(time, min = eps)
    lt <- torch::torch_log(t)

    lt_c <- lt - aft_loc_t

    sigma <- torch::torch_clamp(torch::torch_exp(log_sigma), min = eps)
    z <- (lt_c - mu_resid) / sigma

    logS <- survdnn__log_surv_std_normal(z, eps = eps)

    nll_event <- lt + log_sigma + 0.5 * z^2
    nll_cens  <- -logS

    nll <- torch::torch_where(status == 1, nll_event, nll_cens)
    torch::torch_mean(nll)
  }

  list(
    loss_fn = loss_fn,
    extra_params = list(log_sigma = log_sigma)
  )
}
