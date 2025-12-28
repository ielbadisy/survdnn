#' Predict from a survdnn Model
#'
#' Generate predictions from a fitted `survdnn` model for new data. Supports linear predictors,
#' survival probabilities at specified time points, or cumulative risk estimates.
#'
#' @param object An object of class `"survdnn"` returned by [survdnn()].
#' @param newdata A data frame of new observations to predict on.
#' @param times Numeric vector of time points at which to compute survival or risk probabilities.
#'   Required if `type = "survival"` or `type = "risk"`.
#' @param type Character string specifying the type of prediction to return:
#'   \describe{
#'     \item{"lp"}{Linear predictor (model score).}
#'     \item{"survival"}{Predicted survival probabilities at each value of `times`.}
#'     \item{"risk"}{Cumulative risk (1 - survival) at a single time point.}
#'   }
#' @param ... Currently ignored (for future extensions).
#'
#' @return A numeric vector (if `type = "lp"` or `"risk"`), or a data frame
#'   (if `type = "survival"`) with one row per observation and one column per `times`.
#'
#' @export
predict.survdnn <- function(
  object,
  newdata,
  times = NULL,
  type = c("survival", "lp", "risk"),
  ...
) {

  type <- match.arg(type)

  stopifnot(inherits(object, "survdnn"))
  stopifnot(is.data.frame(newdata))

  loss  <- object$loss
  model <- object$model

  device <- if (!is.null(object$device)) {
    object$device
  } else {
    torch::torch_device("cpu")
  }

  model$to(device = device)
  model$eval()

  ## expand '.' safely using training data
  tt <- stats::terms(object$formula, data = object$data)

  x <- stats::model.matrix(
    stats::delete.response(tt),
    newdata
  )[, object$xnames, drop = FALSE]

  x_scaled <- scale(
    x,
    center = object$x_center,
    scale  = object$x_scale
  )

  ## ================================================================
  ## Cox / Cox L2
  ## ================================================================
  if (loss %in% c("cox", "cox_l2")) {

    x_tensor <- torch::torch_tensor(
      x_scaled,
      dtype  = torch::torch_float(),
      device = device
    )

    torch::with_no_grad({
      lp <- -as.numeric(model(x_tensor)[, 1])
    })

    if (type == "lp") return(lp)

    if (is.null(times)) {
      stop("`times` must be specified for type = 'survival' or 'risk'.", call. = FALSE)
    }
    if (type == "risk" && length(times) != 1) {
      stop("For type = 'risk', `times` must be a single value.", call. = FALSE)
    }

    train_x <- stats::model.matrix(
      stats::delete.response(tt),
      object$data
    )[, object$xnames, drop = FALSE]

    train_x_scaled <- scale(
      train_x,
      center = object$x_center,
      scale  = object$x_scale
    )

    train_x_tensor <- torch::torch_tensor(
      train_x_scaled,
      dtype  = torch::torch_float(),
      device = device
    )

    torch::with_no_grad({
      train_lp <- -as.numeric(model(train_x_tensor)[, 1])
    })

    y_train <- model.response(model.frame(object$formula, object$data))

    train_df <- data.frame(
      time   = y_train[, "time"],
      status = y_train[, "status"],
      lp     = train_lp
    )

    bh <- survival::basehaz(
      survival::coxph(Surv(time, status) ~ lp, data = train_df),
      centered = FALSE
    )

    H0 <- stats::approx(
      bh$time,
      bh$hazard,
      xout = sort(times),
      rule = 2
    )$y

    surv_mat <- outer(
      lp,
      H0,
      function(lp_i, h0_j) exp(-h0_j * exp(lp_i))
    )

    if (type == "risk") {
      return(1 - surv_mat[, length(times)])
    }

    colnames(surv_mat) <- paste0("t=", sort(times))
    return(as.data.frame(surv_mat))
  }

  ## ================================================================
  ## AFT
  ## ================================================================
  if (loss == "aft") {

    x_tensor <- torch::torch_tensor(
      x_scaled,
      dtype  = torch::torch_float(),
      device = device
    )

    torch::with_no_grad({
      pred <- as.numeric(model(x_tensor)[, 1])
    })

    if (type == "lp") return(pred)

    if (is.null(times)) {
      y_train <- model.response(model.frame(object$formula, object$data))
      times <- sort(unique(y_train[, "time"]))
    }

    logt <- log(sort(times))

    surv_mat <- outer(
      pred,
      logt,
      function(fx, lt) 1 - stats::pnorm(lt - fx)
    )

    if (type == "risk") {
      return(1 - surv_mat[, length(times)])
    }

    colnames(surv_mat) <- paste0("t=", sort(times))
    return(as.data.frame(surv_mat))
  }

  ## ================================================================
  ## CoxTime
  ## ================================================================
  if (loss == "coxtime") {

    y_train <- model.response(model.frame(object$formula, object$data))

    time_train   <- y_train[, "time"]
    status_train <- y_train[, "status"]
    event_times  <- sort(unique(time_train[status_train == 1]))

    train_x <- stats::model.matrix(
      stats::delete.response(tt),
      object$data
    )[, object$xnames, drop = FALSE]

    train_x_scaled <- scale(
      train_x,
      center = object$x_center,
      scale  = object$x_scale
    )

    if (length(event_times) == 0) {
      stop("CoxTime prediction requires at least one event in training data.", call. = FALSE)
    }

    ## type = "lp"
    if (type == "lp") {

      t0 <- event_times[1]
      x_temp <- cbind(t0, x_scaled)

      x_tensor <- torch::torch_tensor(
        x_temp,
        dtype  = torch::torch_float(),
        device = device
      )

      torch::with_no_grad({
        lp <- as.numeric(model(x_tensor)[, 1])
      })

      return(lp)
    }

    if (is.null(times)) times <- event_times
    times <- sort(unique(times))

    ## g(T_i, x_new)
    g_new_mat <- matrix(NA_real_, nrow = nrow(x_scaled), ncol = length(event_times))
    for (j in seq_along(event_times)) {
      x_temp <- cbind(event_times[j], x_scaled)
      x_tensor <- torch::torch_tensor(x_temp, dtype = torch::torch_float(), device = device)
      torch::with_no_grad({
        g_new_mat[, j] <- as.numeric(model(x_tensor)[, 1])
      })
    }

    ## g(T_i, x_train)
    g_train_mat <- matrix(NA_real_, nrow = nrow(train_x_scaled), ncol = length(event_times))
    for (j in seq_along(event_times)) {
      x_temp <- cbind(event_times[j], train_x_scaled)
      x_tensor <- torch::torch_tensor(x_temp, dtype = torch::torch_float(), device = device)
      torch::with_no_grad({
        g_train_mat[, j] <- as.numeric(model(x_tensor)[, 1])
      })
    }

    dN    <- as.numeric(table(factor(time_train[status_train == 1], levels = event_times)))
    denom <- colSums(exp(g_train_mat), na.rm = TRUE)
    dH0   <- dN / denom

    H_pred <- matrix(NA_real_, nrow = nrow(g_new_mat), ncol = length(times))
    for (i in seq_along(times)) {
      relevant <- which(event_times <= times[i])
      if (length(relevant) > 0) {
        H_pred[, i] <- rowSums(
          exp(g_new_mat[, relevant, drop = FALSE]) *
            matrix(rep(dH0[relevant], each = nrow(g_new_mat)), nrow = nrow(g_new_mat))
        )
      } else {
        H_pred[, i] <- 0
      }
    }

    S_pred <- exp(-H_pred)

    if (type == "risk") {
      return(1 - S_pred[, length(times)])
    }

    colnames(S_pred) <- paste0("t=", times)
    return(as.data.frame(S_pred))
  }

  ## ================================================================
  ## Royston–Parmar PH / TVE
  ##
  ## Requires: object$loss_state created at fit time by survdnn_make_loss():
  ##   timescale (chr), knots_internal/knot_min/knot_max (torch tensors),
  ##   gamma (torch parameter), and alpha for rp_tve.
  ## ================================================================
  if (loss %in% c("rp_ph", "rp_tve")) {

    if (is.null(object$loss_state) || !is.list(object$loss_state)) {
      stop("RP prediction requires `object$loss_state` created at fit time.", call. = FALSE)
    }

    st <- object$loss_state

    needed <- c("timescale", "knots_internal", "knot_min", "knot_max", "gamma")
    missing <- setdiff(needed, names(st))
    if (length(missing) > 0) {
      stop("RP prediction: missing loss_state fields: ", paste(missing, collapse = ", "), call. = FALSE)
    }
    if (loss == "rp_tve" && (is.null(st$alpha) || !inherits(st$alpha, "torch_tensor"))) {
      stop("RP-TVE prediction: missing `loss_state$alpha`.", call. = FALSE)
    }

    timescale <- st$timescale
    if (!timescale %in% c("log", "identity")) {
      stop("RP prediction: invalid `loss_state$timescale`.", call. = FALSE)
    }

    if (type != "lp") {
      if (is.null(times)) {
        stop("`times` must be specified for type = 'survival' or 'risk'.", call. = FALSE)
      }
      if (type == "risk" && length(times) != 1) {
        stop("For type = 'risk', `times` must be a single value.", call. = FALSE)
      }
    }

    # f_theta(x) as torch tensor [n]
    x_tensor <- torch::torch_tensor(
      x_scaled,
      dtype  = torch::torch_float(),
      device = device
    )

    torch::with_no_grad({
      fx <- model(x_tensor)[, 1]  # [n]
    })

    if (type == "lp") {
      return(as.numeric(fx$to(device = "cpu")))
    }

    times <- sort(unique(as.numeric(times)))
    if (any(!is.finite(times)) || any(times <= 0)) {
      stop("`times` must be finite and > 0.", call. = FALSE)
    }

    # move state to current device
    knots_internal <- st$knots_internal$to(device = device, dtype = torch::torch_float())
    knot_min       <- st$knot_min$to(device = device, dtype = torch::torch_float())
    knot_max       <- st$knot_max$to(device = device, dtype = torch::torch_float())
    gamma          <- st$gamma$to(device = device, dtype = torch::torch_float())
    alpha          <- if (loss == "rp_tve") st$alpha$to(device = device, dtype = torch::torch_float()) else NULL

    # time grid + spline basis
    t_tensor <- torch::torch_tensor(times, dtype = torch::torch_float(), device = device)
    z <- if (timescale == "log") t_tensor$log() else t_tensor

    bd <- torch_rcs_basis(
      z,
      knots_internal = knots_internal,
      knot_min = knot_min,
      knot_max = knot_max
    )
    B <- bd$B  # [m, p]

    # eta0(t) = B %*% gamma
    g <- gamma$view(c(-1))
    torch::with_no_grad({
      eta0 <- torch::torch_matmul(B, g$unsqueeze(2))$view(c(-1))  # [m]
    })

    # broadcast to [n, m]
    eta0_row <- eta0$unsqueeze(1)  # [1, m]
    fx_col   <- fx$unsqueeze(2)    # [n, 1]

    if (loss == "rp_ph") {

      torch::with_no_grad({
        eta_nm <- fx_col + eta0_row      # [n, m]
        H_nm   <- eta_nm$exp()
        S_nm   <- (-H_nm)$exp()
      })

    } else {

      a <- alpha$view(c(-1))
      torch::with_no_grad({
        s1 <- torch::torch_matmul(B, a$unsqueeze(2))$view(c(-1))  # [m]
        s1_row <- s1$unsqueeze(1)                                 # [1, m]

        eta_nm <- eta0_row + fx_col * s1_row
        H_nm   <- eta_nm$exp()
        S_nm   <- (-H_nm)$exp()
      })
    }

    S_mat <- as.matrix(S_nm$to(device = torch::torch_device("cpu")))

    if (type == "risk") {
      return(1 - S_mat[, length(times)])
    }

    colnames(S_mat) <- paste0("t=", times)
    return(as.data.frame(S_mat))
  }

  stop("Unsupported loss type for prediction: ", loss, call. = FALSE)
}
