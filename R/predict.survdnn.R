#' Predict from a survdnn Model
#'
#' Generate predictions from a fitted `survdnn` model for new data. Supports linear predictors,
#' survival probabilities at specified time points, or cumulative risk estimates.
#'
#' @param object An object of class `"survdnn"` returned by [survdnn()].
#' @param newdata A data frame of new observations to predict on.
#' @param times Numeric vector of time points at which to compute survival or risk probabilities.
#'   Required if `type = "survival"` or `type = "risk"` for Cox/AFT models.
#'   For CoxTime, `times = NULL` is allowed when `type="survival"` and defaults to event times.
#' @param type Character string specifying the type of prediction to return:
#'   \describe{
#'     \item{"lp"}{Linear predictor. For `"cox"`/`"cox_l2"` this is a log-risk score
#'     (higher implies worse prognosis, consistent with training sign convention). For `"aft"`,
#'     this is the predicted location parameter \eqn{\mu(x)} on the log-time scale. For `"coxtime"`,
#'     this is \eqn{g(t_0, x)} evaluated at a reference time \eqn{t_0} (the first event time).}
#'     \item{"survival"}{Predicted survival probabilities at each value of `times`.}
#'     \item{"risk"}{Cumulative risk (1 - survival) at **a single** time point.}
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

  device <- if (!is.null(object$device)) object$device else torch::torch_device("cpu")
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

  ## IMPORTANT: type='risk' is defined at a single time point
  if (type == "risk" && !is.null(times) && length(times) != 1) {
    stop("For type = 'risk', `times` must be a single numeric value.", call. = FALSE)
  }

  ## Cox / Cox L2
  if (loss %in% c("cox", "cox_l2")) {

    if (type %in% c("survival", "risk") && is.null(times)) {
      stop("`times` must be specified for type = 'survival' or 'risk'.", call. = FALSE)
    }

    x_tensor <- torch::torch_tensor(
      x_scaled,
      dtype  = torch::torch_float(),
      device = device
    )

    torch::with_no_grad({
      lp <- -as.numeric(model(x_tensor)[, 1])
    })

    if (type == "lp") return(lp)

    ## baseline hazard via Breslow on training data
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
      survival::coxph(survival::Surv(time, status) ~ lp, data = train_df),
      centered = FALSE
    )

    times_sorted <- sort(as.numeric(times))
    H0 <- stats::approx(bh$time, bh$hazard, xout = times_sorted, rule = 2)$y

    surv_mat <- outer(
      lp,
      H0,
      function(lp_i, h0_j) exp(-h0_j * exp(lp_i))
    )

    if (type == "risk") return(1 - surv_mat[, 1])

    colnames(surv_mat) <- paste0("t=", times_sorted)
    return(as.data.frame(surv_mat))
  }

  ## AFT (log-normal AFT with learned global sigma + training centering)
  if (loss == "aft") {

    if (type %in% c("survival", "risk") && is.null(times)) {
      stop("`times` must be specified for type = 'survival' or 'risk'.", call. = FALSE)
    }

    x_tensor <- torch::torch_tensor(
      x_scaled,
      dtype  = torch::torch_float(),
      device = device
    )

    torch::with_no_grad({
      mu_raw <- as.numeric(model(x_tensor)[, 1])
    })

    loc <- if (!is.null(object$aft_loc) && is.finite(object$aft_loc)) object$aft_loc else 0
    mu  <- mu_raw + loc
    if (type == "lp") return(mu)

    ## sigma: must be finite and > 0; otherwise fall back to 1
    ls <- object$aft_log_sigma
    sigma <- if (!is.null(ls) && is.finite(ls)) exp(ls) else 1
    if (!is.finite(sigma) || sigma <= 0) sigma <- 1

    times_sorted <- sort(as.numeric(times))
    times_sorted[times_sorted <= 0] <- .Machine$double.eps
    logt <- log(times_sorted)

    surv_mat <- outer(
      mu,
      logt,
      function(mu_i, lt) 1 - stats::pnorm((lt - mu_i) / sigma)
    )

    surv_mat[surv_mat < 0] <- 0
    surv_mat[surv_mat > 1] <- 1

    if (type == "risk") return(1 - surv_mat[, 1])

    colnames(surv_mat) <- paste0("t=", times_sorted)
    return(as.data.frame(surv_mat))
  }

  ## CoxTime
  if (loss == "coxtime") {

    y_train <- model.response(model.frame(object$formula, object$data))
    time_train   <- y_train[, "time"]
    status_train <- y_train[, "status"]
    event_times  <- sort(unique(time_train[status_train == 1]))

    if (length(event_times) == 0) {
      stop("CoxTime prediction requires at least one event in training data.", call. = FALSE)
    }

    ## --- time scaling used in training (fallback-safe) ---
    t_center <- if (!is.null(object$coxtime_time_center) && is.finite(object$coxtime_time_center)) {
      as.numeric(object$coxtime_time_center)
    } else 0

    t_scale <- if (!is.null(object$coxtime_time_scale) && is.finite(object$coxtime_time_scale) &&
                  as.numeric(object$coxtime_time_scale) > 0) {
      as.numeric(object$coxtime_time_scale)
    } else 1

    scale_t <- function(t) (as.numeric(t) - t_center) / t_scale

    ## training covariates (scaled) for baseline computation
    train_x <- stats::model.matrix(
      stats::delete.response(tt),
      object$data
    )[, object$xnames, drop = FALSE]

    train_x_scaled <- scale(
      train_x,
      center = object$x_center,
      scale  = object$x_scale
    )

    ## type = "lp": define lp at a reference time (first event time)
    if (type == "lp") {
      t0 <- event_times[1]
      t0s <- scale_t(t0)
      x_temp <- cbind(t0s, x_scaled)
      x_tensor <- torch::torch_tensor(x_temp, dtype = torch::torch_float(), device = device)
      torch::with_no_grad({
        lp <- as.numeric(model(x_tensor)[, 1])
      })
      return(lp)
    }

    ## For CoxTime: allow times=NULL for survival -> default event_times (RAW)
    if (type == "survival" && is.null(times)) {
      times_sorted <- event_times
    } else {
      if (type %in% c("survival", "risk") && is.null(times)) {
        stop("`times` must be specified for type = 'survival' or 'risk'.", call. = FALSE)
      }
      times_sorted <- sort(unique(as.numeric(times)))
    }

    ## Compute g(t_k, x_new) on event-time grid
    ## NOTE: net expects SCALED time input
    g_new_mat <- matrix(NA_real_, nrow = nrow(x_scaled), ncol = length(event_times))
    for (j in seq_along(event_times)) {
      tj  <- event_times[j]
      tjs <- scale_t(tj)
      x_temp <- cbind(tjs, x_scaled)
      x_tensor <- torch::torch_tensor(x_temp, dtype = torch::torch_float(), device = device)
      torch::with_no_grad({
        g_new_mat[, j] <- as.numeric(model(x_tensor)[, 1])
      })
    }

    ## Compute g(t_k, x_train) on event-time grid (scaled time input)
    g_train_mat <- matrix(NA_real_, nrow = nrow(train_x_scaled), ncol = length(event_times))
    for (j in seq_along(event_times)) {
      tj  <- event_times[j]
      tjs <- scale_t(tj)
      x_temp <- cbind(tjs, train_x_scaled)
      x_tensor <- torch::torch_tensor(x_temp, dtype = torch::torch_float(), device = device)
      torch::with_no_grad({
        g_train_mat[, j] <- as.numeric(model(x_tensor)[, 1])
      })
    }

    ## Baseline increments: dH0(t_k) = dN(t_k) / sum_{j in R(t_k)} exp(g(t_k, x_j))
    ## risk sets are defined on RAW time (a proxy for risk intensity)
    dN <- as.numeric(table(factor(time_train[status_train == 1], levels = event_times)))

    risk_mat <- outer(time_train, event_times, `>=`)
    denom <- colSums(exp(g_train_mat) * risk_mat, na.rm = TRUE)

    denom[denom <= 0] <- NA_real_
    dH0 <- dN / denom
    dH0[is.na(dH0)] <- 0

    ## Cumulative hazard at requested times (RAW time grid)
    H_pred <- matrix(0, nrow = nrow(g_new_mat), ncol = length(times_sorted))
    for (i in seq_along(times_sorted)) {
      relevant <- which(event_times <= times_sorted[i])
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
    S_pred[S_pred < 0] <- 0
    S_pred[S_pred > 1] <- 1

    if (type == "risk") return(1 - S_pred[, 1])

    colnames(S_pred) <- paste0("t=", times_sorted)
    return(as.data.frame(S_pred))
  }

  stop("Unsupported loss type for prediction: ", loss, call. = FALSE)
}
