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
#'     \item{"lp"}{Linear predictor (log-risk score; higher implies worse prognosis).}
#'     \item{"survival"}{Predicted survival probabilities at each value of `times`.}
#'     \item{"risk"}{Cumulative risk (1 - survival) at a single time point.}
#'   }
#' @param ... Currently ignored (for future extensions).
#'
#' @return A numeric vector (if `type = "lp"` or `"risk"`), or a data frame (if `type = "survival"`)
#'   with one row per observation and one column per `times`.
#'
#' @export
#'
#' @examples
#' \donttest{
#' library(survival)
#' data(veteran, package = "survival")
#'
#' # Fit survdnn with Cox loss
#' mod <- survdnn(Surv(time, status) ~ age + karno + celltype, data = veteran,
#'                loss = "cox", epochs = 50, verbose = FALSE)
#'
#' # Linear predictor (log-risk)
#' predict(mod, newdata = veteran, type = "lp")[1:5]
#'
#' # Survival probabilities at selected times
#' predict(mod, newdata = veteran, type = "survival", times = c(30, 90, 180))[1:5, ]
#'
#' # Cumulative risk at 180 days
#' predict(mod, newdata = veteran, type = "risk", times = 180)[1:5]
#' }

predict.survdnn <- function(object, newdata, times = NULL,
                            type = c("survival", "lp", "risk"), ...) {
  type <- match.arg(type)
  stopifnot(inherits(object, "survdnn"))
  stopifnot(is.data.frame(newdata))

  loss <- object$loss
  model <- object$model
  model$eval()

  x <- model.matrix(delete.response(terms(object$formula)), newdata)[, object$xnames, drop = FALSE]
  x_scaled <- scale(x, center = object$x_center, scale = object$x_scale)

  ## cox 1 cox_l2
  if (loss %in% c("cox", "cox_l2")) {
    x_tensor <- torch::torch_tensor(x_scaled, dtype = torch::torch_float())
    torch::with_no_grad({ lp <- -as.numeric(model(x_tensor)[, 1]) })
    if (type == "lp") return(lp)
    if (is.null(times)) stop("`times` must be specified for type = 'survival' or 'risk'.")
    if (type == "risk" && length(times) != 1) stop("For type = 'risk', `times` must be a single value.")

    train_x <- model.matrix(delete.response(terms(object$formula)), object$data)[, object$xnames, drop = FALSE]
    train_x_scaled <- scale(train_x, center = object$x_center, scale = object$x_scale)
    train_lp <- -as.numeric(model(torch::torch_tensor(train_x_scaled, dtype = torch::torch_float()))[, 1])
    y_train <- model.response(model.frame(object$formula, object$data))
    train_df <- data.frame(time = y_train[, "time"], status = y_train[, "status"], lp = train_lp)
    bh <- survival::basehaz(survival::coxph(Surv(time, status) ~ lp, data = train_df), centered = FALSE)
    H0 <- approx(bh$time, bh$hazard, xout = sort(times), rule = 2)$y
    surv_mat <- outer(lp, H0, function(lp_i, h0_j) exp(-h0_j * exp(lp_i)))
    if (type == "risk") return(1 - surv_mat[, length(times)])
    colnames(surv_mat) <- paste0("t=", sort(times))
    return(as.data.frame(surv_mat))
  }

  ## AFT
  if (loss == "aft") {
    x_tensor <- torch::torch_tensor(x_scaled, dtype = torch::torch_float())
    torch::with_no_grad({ pred <- as.numeric(model(x_tensor)[, 1]) })
    if (type == "lp") return(pred)
    if (is.null(times)) {
      y_train <- model.response(model.frame(object$formula, object$data))
      times <- sort(unique(y_train[, "time"]))
    }
    logt <- log(sort(times))
    surv_mat <- outer(pred, logt, function(fx, lt) 1 - pnorm(lt - fx))
    if (type == "risk") return(1 - surv_mat[, length(times)])
    colnames(surv_mat) <- paste0("t=", sort(times))
    return(as.data.frame(surv_mat))
  }

  ## coxtime (deephit)

  if (loss == "coxtime") {
    y_train <- model.response(model.frame(object$formula, object$data))
    time_train <- y_train[, "time"]
    status_train <- y_train[, "status"]
    event_times <- sort(unique(time_train[status_train == 1]))

    train_x <- model.matrix(delete.response(terms(object$formula)), object$data)[, object$xnames, drop = FALSE]
    train_x_scaled <- scale(train_x, center = object$x_center, scale = object$x_scale)

    if (is.null(times)) times <- event_times
    times <- sort(unique(times))

    # evaluate g(Ti, x_new)
    g_new_mat <- matrix(NA, nrow = nrow(x_scaled), ncol = length(event_times))
    for (j in seq_along(event_times)) {
      x_temp <- cbind(event_times[j], x_scaled)
      x_tensor <- torch::torch_tensor(x_temp, dtype = torch::torch_float())
      torch::with_no_grad({ g_new_mat[, j] <- as.numeric(model(x_tensor)[, 1]) })
    }

    # evaluate g(Ti, x_train)
    g_train_mat <- matrix(NA, nrow = nrow(train_x_scaled), ncol = length(event_times))
    for (j in seq_along(event_times)) {
      x_temp <- cbind(event_times[j], train_x_scaled)
      x_tensor <- torch::torch_tensor(x_temp, dtype = torch::torch_float())
      torch::with_no_grad({ g_train_mat[, j] <- as.numeric(model(x_tensor)[, 1]) })
    }

    dN <- as.numeric(table(factor(time_train[status_train == 1], levels = event_times)))
    denom <- colSums(exp(g_train_mat), na.rm = TRUE)
    dH0 <- dN / denom

    H_pred <- matrix(NA, nrow = nrow(g_new_mat), ncol = length(times))
    for (i in seq_along(times)) {
      relevant <- which(event_times <= times[i])
      if (length(relevant) > 0) {
        H_pred[, i] <- rowSums(exp(g_new_mat[, relevant, drop = FALSE]) * matrix(rep(dH0[relevant], each = nrow(g_new_mat)), nrow = nrow(g_new_mat)))
      } else {
        H_pred[, i] <- 0
      }
    }

    S_pred <- exp(-H_pred)
    if (type == "risk") return(1 - S_pred[, length(times)])
    colnames(S_pred) <- paste0("t=", times)
    return(as.data.frame(S_pred))
  }

  stop("Unsupported loss type for prediction: ", loss)
}
