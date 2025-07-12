#' Predict from a survdnn Model
#'
#' Generate predictions from a fitted `survdnn` model for new data. Supports linear predictors,
#' survival probabilities at specified time points, or risk estimates at a single time point.
#'
#' @param object An object of class `"survdnn"` returned by [survdnn()].
#' @param newdata A data frame of new observations to predict on.
#' @param times Numeric vector of time points at which to compute survival or risk probabilities.
#'   Required if `type = "survival"` or `type = "risk"`.
#' @param type Character string specifying the type of prediction to return:
#'   \describe{
#'     \item{"lp"}{Linear predictor (log-risk score, higher = worse prognosis).}
#'     \item{"survival"}{Survival probabilities at `times`.}
#'     \item{"risk"}{Cumulative risk (1 - survival) at a single time point.}
#'   }
#' @param ... Currently ignored (for future compatibility).
#'
#' @return A numeric vector (if `type = "lp"` or `"risk"`), or a data frame with survival
#' probabilities (if `type = "survival"`).
#'
#' @export
#'
#' @examples
#' library(survival)
#' data(veteran, package = "survival")
#' mod <- survdnn(Surv(time, status) ~ age + karno + celltype, data = veteran, epochs = 50, verbose = FALSE)
#' predict(mod, newdata = veteran, type = "lp")[1:5]
#' predict(mod, newdata = veteran, type = "survival", times = c(30, 90, 180))[1:5, ]
#' predict(mod, newdata = veteran, type = "risk", times = 180)[1:5]


predict.survdnn <- function(object, newdata, times = NULL,
                            type = c("survival", "lp", "risk"), ...) {
  type <- match.arg(type)
  stopifnot(inherits(object, "survdnn"))
  stopifnot(is.data.frame(newdata))

  # rebuild predictor matrix
  x <- model.matrix(delete.response(terms(object$formula)), newdata)[, object$xnames, drop = FALSE]
  x_scaled <- scale(x, center = object$x_center, scale = object$x_scale)
  x_tensor <- torch::torch_tensor(as.matrix(x_scaled), dtype = torch::torch_float())

  # predict linear predictor (negative risk score)
  object$model$eval()

  ## torch inference is done inside with_no_grad()
  ## model outputs genative risk: we flip sign to get cox style linear predictor
  with_no_grad({lp <- -as.numeric(object$model(x_tensor)[, 1])})

  if (type == "lp") return(lp)
  ## times handling for lp
  if (is.null(times)) stop("`times` must be specified for type = 'survival' or 'risk'.")
  times <- sort(times)
  ## add check for length(times) == 1 when type = "risk"
  if (type == "risk" && length(times) != 1) {
    stop("For type = 'risk', `times` must be a single numeric value.")
  }
  
  # estimate baseline hazard using training set
  train_x <- model.matrix(delete.response(terms(object$formula)), object$data)[, object$xnames, drop = FALSE]
  train_x_scaled <- scale(train_x, center = object$x_center, scale = object$x_scale)
  train_lp <- -as.numeric(object$model(torch::torch_tensor(train_x_scaled, dtype = torch::torch_float()))[, 1])
  train_df <- data.frame(
    time = model.response(model.frame(object$formula, object$data))[, "time"],
    status = model.response(model.frame(object$formula, object$data))[, "status"],
    lp = train_lp
  )
  bh <- survival::basehaz(survival::coxph(Surv(time, status) ~ lp, data = train_df), centered = FALSE)
  H0 <- approx(bh$time, bh$hazard, xout = times, rule = 2)$y

  # compute survival matrix
  lp_clipped <- pmin(pmax(lp, -3), 3)
  surv_mat <- outer(lp_clipped, H0, function(lp_i, h0_j) exp(-h0_j * exp(lp_i)))

  if (type == "risk") {
    return(1 - surv_mat[, length(times)])
  }

  surv_df <- as.data.frame(surv_mat)
  colnames(surv_df) <- paste0("t=", times)
  return(surv_df)
}

