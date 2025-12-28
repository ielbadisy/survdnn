#' Loss Functions for survdnn Models
#'
#' These functions define various loss functions used internally by `survdnn()` for training deep neural networks on right-censored survival data.
#'
#' @section Supported Losses:
#' - **Cox partial likelihood loss** (`cox_loss`): Negative partial log-likelihood used in proportional hazards modeling.
#' - **L2-penalized Cox loss** (`cox_l2_loss`): Adds L2 regularization to the Cox loss.
#' - **Accelerated Failure Time (AFT) loss** (`aft_loss`): Mean squared error between predicted and log-transformed event times, applied to uncensored observations only.
#' - **CoxTime loss** (`coxtime_loss`): Partial likelihood loss from Kvamme & Borgan (2019) for Cox-Time models.
#' - **Royston–Parmar PH loss** (`rp_ph_loss`): Full log-likelihood for a flexible parametric baseline on the log cumulative hazard (PH) scale, combined with a DNN risk score.
#' - **Royston–Parmar TVE loss** (`rp_tve_loss`): Extends RP-PH with a single-component time-varying effect while keeping `output_dim = 1`, modeling
#'   \deqn{\eta(t \mid x) = \eta_0(t) + f_\theta(x)\, s_1(t).}
#'
#' @param pred A tensor of predicted values (typically linear predictors or log-times).
#' @param true A tensor with two columns: observed time and status (1 = event, 0 = censored).
#' @param lambda Regularization parameter for `cox_l2_loss` (default: `1e-4`).
#' @param gamma Torch parameter vector for the RP baseline spline coefficients.
#' @param alpha Torch parameter vector for the RP time-varying effect spline coefficients (TVE only).
#' @param knots_internal Torch tensor of internal knot locations on the chosen time scale.
#' @param knot_min,knot_max Torch scalars for boundary knots on the chosen time scale.
#' @param timescale Character. Time scale used in the spline basis: `"log"` uses `z = log(t)`; `"identity"` uses `z = t`.
#' @param eps_time Small positive constant to clamp time away from zero for numerical stability.
#' @param eps_h Small positive constant to clamp hazard away from zero for numerical stability.
#' @param mono_penalty Nonnegative penalty weight to discourage non-increasing cumulative hazard (i.e., negative derivative).
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
  status <- status[idx]
  pred <- -pred[idx, 1]  # negate for log-partial likelihood

  log_cumsum_exp <- torch_logcumsumexp(pred, dim = 1)
  event_mask <- (status == 1)

  -torch_mean(pred[event_mask] - log_cumsum_exp[event_mask])
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
    # keep dtype/device consistent
    return(torch_zeros_like(pred[1, 1]))
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

  # risk set indicator: R_ij = 1 if time_j >= time_i
  time_i <- time$view(c(n, 1))  # [n,1]
  time_j <- time$view(c(1, n))  # [1,n]
  risk_bool <- (time_j >= time_i)  # [n,n] bool

  # difference: g(t_i, x_j) - g(t_i, x_i)
  pred_i <- pred$view(c(n, 1))  # [n,1]
  pred_j <- pred$view(c(1, n))  # [1,n]
  diff <- pred_j - pred_i       # [n,n]

  event_mask <- (status == 1)

  # exact masking: set non-risk to a very negative number before logsumexp
  neg_inf <- torch::torch_tensor(-1e30, dtype = pred$dtype, device = pred$device)
  masked <- torch_where(risk_bool, diff, neg_inf)
  log_sum_exp <- torch_logsumexp(masked, dim = 2)  # [n]

  torch_mean(log_sum_exp[event_mask])
}


# ==============================================================
# Restricted cubic spline (RCS) basis on z with analytic derivative
# Harrell-style RCS with boundary knots knot_min, knot_max:
#   B  : [n, K+2] = (1, z, b1..bK)
#   dB : [n, K+2] = d/dz of the above
# Inputs are torch tensors (GPU-safe)
# ==============================================================

torch_rcs_basis <- function(z, knots_internal, knot_min, knot_max) {
  z <- z$view(c(-1))
  n <- z$size()[[1]]

  one  <- torch_ones_like(z)
  zero <- torch_zeros_like(z)

  # No internal knots -> linear spline: (1, z)
  if (as.integer(knots_internal$numel()) == 0L) {
    B  <- torch_stack(list(one, z), dim = 2)
    dB <- torch_stack(list(zero, one), dim = 2)
    return(list(B = B, dB = dB))
  }

  K <- as.integer(knots_internal$size()[[1]])

  pospow <- function(u, p) torch_relu(u)$pow(p)

  km <- knot_min
  kM <- knot_max

  # Broadcast z (n,1) and internal knots (1,K)
  z_mat <- z$unsqueeze(2)                  # [n,1]
  k_mat <- knots_internal$unsqueeze(1)$t() # [1,K]

  zk3 <- pospow(z_mat - k_mat, 3)          # [n,K]
  zk2 <- pospow(z_mat - k_mat, 2)          # [n,K]

  zkm3 <- pospow(z - km, 3)$unsqueeze(2)$expand(c(n, K))
  zkm2 <- pospow(z - km, 2)$unsqueeze(2)$expand(c(n, K))

  zkM3 <- pospow(z - kM, 3)$unsqueeze(2)$expand(c(n, K))
  zkM2 <- pospow(z - kM, 2)$unsqueeze(2)$expand(c(n, K))

  denom <- (kM - km)$clamp_min(1e-12)

  wM <- (kM - knots_internal) / denom      # [K]
  wm <- (km - knots_internal) / denom      # [K]

  wM_mat <- wM$unsqueeze(1)$t()            # [1,K]
  wm_mat <- wm$unsqueeze(1)$t()            # [1,K]

  b  <- zk3 - zkM3 * wM_mat + zkm3 * wm_mat
  db <- 3 * zk2 - 3 * zkM2 * wM_mat + 3 * zkm2 * wm_mat

  B  <- torch_cat(list(one$unsqueeze(2), z$unsqueeze(2), b),  dim = 2)
  dB <- torch_cat(list(zero$unsqueeze(2), one$unsqueeze(2), db), dim = 2)

  list(B = B, dB = dB)
}


# ==============================================================
# Royston–Parmar losses (PH scale) with optional monotonicity penalty
# eta(t|x) = eta0(t) + f(x)             (rp_ph)
# eta(t|x) = eta0(t) + f(x) * s1(t)     (rp_tve, single-component TVE)
# where eta0(t) = B(z)^T gamma, s1(t) = B(z)^T alpha, z=log(t) or z=t
# ==============================================================

#' @rdname survdnn_losses
#' @export
rp_ph_loss <- function(pred, true,
                       gamma,
                       knots_internal, knot_min, knot_max,
                       timescale = c("log", "identity"),
                       eps_time = 1e-12, eps_h = 1e-12,
                       mono_penalty = 1e-2) {

  timescale <- match.arg(timescale)

  time   <- true[, 1]
  status <- true[, 2]

  t  <- time$to(dtype = pred$dtype)$clamp_min(eps_time)$view(c(-1))
  d  <- status$to(dtype = pred$dtype)$view(c(-1))
  fx <- pred$view(c(-1))

  z <- if (timescale == "log") t$log() else t

  bd <- torch_rcs_basis(z, knots_internal, knot_min, knot_max)
  B  <- bd$B
  dB <- bd$dB

  g <- gamma$view(c(-1))

  eta0     <- torch::torch_matmul(B,  g$unsqueeze(2))$view(c(-1))
  deta0_dz <- torch::torch_matmul(dB, g$unsqueeze(2))$view(c(-1))

  eta <- eta0 + fx
  H   <- eta$exp()

  dz_dt   <- if (timescale == "log") (1 / t) else torch_ones_like(t)
  deta_dt <- deta0_dz * dz_dt

  h_raw <- H * deta_dt
  h     <- h_raw$clamp_min(eps_h)

  ll <- torch_sum(d * torch_log(h) - H)
  n  <- t$size()[[1]]

  pen <- torch_mean(torch_relu(-deta_dt))
  (-ll / n) + mono_penalty * pen
}


#' @rdname survdnn_losses
#' @export
rp_tve_loss <- function(pred, true,
                        gamma, alpha,
                        knots_internal, knot_min, knot_max,
                        timescale = c("log", "identity"),
                        eps_time = 1e-12, eps_h = 1e-12,
                        mono_penalty = 1e-2) {

  timescale <- match.arg(timescale)

  time   <- true[, 1]
  status <- true[, 2]

  t  <- time$to(dtype = pred$dtype)$clamp_min(eps_time)$view(c(-1))
  d  <- status$to(dtype = pred$dtype)$view(c(-1))
  fx <- pred$view(c(-1))

  z <- if (timescale == "log") t$log() else t

  bd <- torch_rcs_basis(z, knots_internal, knot_min, knot_max)
  B  <- bd$B
  dB <- bd$dB

  g <- gamma$view(c(-1))
  a <- alpha$view(c(-1))

  eta0     <- torch::torch_matmul(B,  g$unsqueeze(2))$view(c(-1))
  deta0_dz <- torch::torch_matmul(dB, g$unsqueeze(2))$view(c(-1))

  s1     <- torch::torch_matmul(B,  a$unsqueeze(2))$view(c(-1))
  ds1_dz <- torch::torch_matmul(dB, a$unsqueeze(2))$view(c(-1))

  eta <- eta0 + fx * s1
  H   <- eta$exp()

  dz_dt   <- if (timescale == "log") (1 / t) else torch_ones_like(t)
  deta_dt <- (deta0_dz + fx * ds1_dz) * dz_dt

  h_raw <- H * deta_dt
  h     <- h_raw$clamp_min(eps_h)

  ll <- torch_sum(d * torch_log(h) - H)
  n  <- t$size()[[1]]

  pen <- torch_mean(torch_relu(-deta_dt))
  (-ll / n) + mono_penalty * pen
}


# ==============================================================
# Internal loss factory (standardized)
# Returns: list(fn = function(pred,true), params = list(...))
# ==============================================================

survdnn_make_loss <- function(loss, y_tensor, device) {
  loss <- match.arg(loss, c("cox", "cox_l2", "aft", "coxtime", "rp_ph", "rp_tve"))

  if (loss == "cox") {
    return(list(fn = cox_loss, params = list(), state = NULL))
  }

  if (loss == "aft") {
    return(list(fn = aft_loss, params = list(), state = NULL))
  }

  if (loss == "coxtime") {
    return(list(fn = coxtime_loss, params = list(), state = NULL))
  }

  if (loss == "cox_l2") {
    return(list(
      fn = function(pred, true) cox_l2_loss(pred, true, lambda = 1e-3),
      params = list(),
      state = NULL
    ))
  }

  # ------------------------------------------------------------
  # RP losses: all hyperparameters + trainable parameters live here
  # ------------------------------------------------------------
  k_internal_default <- 1L
  timescale          <- "log"
  mono_penalty       <- 1e-2

  time   <- y_tensor[, 1]
  status <- y_tensor[, 2]

  ev <- (status == 1)
  n_ev <- as.integer(torch_sum(ev)$item())
  if (n_ev == 0L) {
    stop("RP losses require at least one event in the training data.", call. = FALSE)
  }

  # event times on CPU for knot selection (one-time)
  t_ev <- as.numeric(time[ev]$to(device = torch::torch_device("cpu")))
  if (any(!is.finite(t_ev)) || any(t_ev <= 0)) {
    stop("RP losses require strictly positive, finite event times.", call. = FALSE)
  }

  z_ev <- if (timescale == "log") log(t_ev) else t_ev
  uq   <- sort(unique(z_ev))
  if (length(uq) < 2L) {
    stop("RP losses require at least two distinct event times.", call. = FALSE)
  }

  max_k <- max(0L, length(uq) - 2L)
  k_use <- min(as.integer(k_internal_default), max_k)

  iknots <- numeric(0)
  if (k_use > 0L) {
    probs  <- seq(0, 1, length.out = k_use + 2L)[-c(1, k_use + 2L)]
    iknots <- as.numeric(stats::quantile(z_ev, probs = probs, type = 7, names = FALSE))
    iknots <- sort(unique(iknots))
  }

  knot_min_val <- min(uq)
  knot_max_val <- max(uq)

  # store knots as torch tensors on device (GPU-safe)
  knots_internal <- torch::torch_tensor(iknots, dtype = torch::torch_float(), device = device)
  knot_min       <- torch::torch_tensor(knot_min_val, dtype = torch::torch_float(), device = device)
  knot_max       <- torch::torch_tensor(knot_max_val, dtype = torch::torch_float(), device = device)

  basis_dim <- length(iknots) + 2L

  gamma0 <- torch::torch_zeros(basis_dim, dtype = torch::torch_float(), device = device)
  if (basis_dim >= 2L) gamma0[2] <- 0.1
  gamma <- torch::nn_parameter(gamma0)

  # state that must be saved in the fitted model for prediction
  state <- list(
    timescale      = timescale,
    knots_internal = knots_internal,
    knot_min       = knot_min,
    knot_max       = knot_max,
    gamma          = gamma
  )

  if (loss == "rp_ph") {
    fn <- function(pred, true) {
      rp_ph_loss(
        pred = pred, true = true,
        gamma = gamma,
        knots_internal = knots_internal,
        knot_min = knot_min, knot_max = knot_max,
        timescale = timescale,
        mono_penalty = mono_penalty
      )
    }
    return(list(fn = fn, params = list(gamma), state = state))
  }

  alpha0 <- torch::torch_zeros(basis_dim, dtype = torch::torch_float(), device = device)
  alpha  <- torch::nn_parameter(alpha0)

  state$alpha <- alpha

  fn <- function(pred, true) {
    rp_tve_loss(
      pred = pred, true = true,
      gamma = gamma, alpha = alpha,
      knots_internal = knots_internal,
      knot_min = knot_min, knot_max = knot_max,
      timescale = timescale,
      mono_penalty = mono_penalty
    )
  }

  list(fn = fn, params = list(gamma, alpha), state = state)
}
