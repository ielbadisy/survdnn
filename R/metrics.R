#' Concordance Index from a Survival Probability Matrix
#'
#' Computes the time-dependent concordance index (C-index) from a predicted survival matrix
#' at a fixed time point. The risk is computed as `1 - S(t_star)`.
#'
#' @param object A `Surv` object representing the observed survival data.
#' @param predicted A data frame or matrix of predicted survival probabilities.
#'   Each column corresponds to a time point (e.g., `t=90`, `t=180`).
#' @param t_star A numeric time point corresponding to one of the columns in `predicted`.
#'   If `NULL`, the last column is used.
#'
#' @return A single numeric value representing the C-index.
#' @export
#'
#' @examples
#' \donttest{
#' library(survival)
#' data(veteran, package = "survival")
#' mod <- survdnn(Surv(time, status) ~
#' age + karno + celltype, data = veteran, epochs = 50, verbose = FALSE)
#' pred <- predict(mod, newdata = veteran, type = "survival", times = c(30, 90, 180))
#' y <- model.response(model.frame(mod$formula, veteran))
#' cindex_survmat(y, pred, t_star = 180)
#' }

cindex_survmat <- function(object, predicted, t_star = NULL) {
  if (!inherits(object, "Surv")) stop("object must be a survival object (from Surv())")

  time <- object[, 1]
  status <- object[, 2]

  if (!is.null(t_star)) {
    t_name <- paste0("t=", t_star)
    if (!(t_name %in% colnames(predicted))) {
      stop("t_star = ", t_star, " not found in predicted survival matrix.")
    }
    surv_prob <- predicted[[t_name]]
  } else {
    surv_prob <- predicted[[ncol(predicted)]]
  }

  risk_score <- 1 - surv_prob
  n <- length(time)

  concord <- 0
  par_concord <- 0
  permissible <- 0

  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      if ((time[i] < time[j] && status[i] == 0) ||
          (time[j] < time[i] && status[j] == 0)) next
      if (time[i] == time[j] && status[i] == 0 && status[j] == 0) next

      permissible <- permissible + 1

      if (time[i] != time[j]) {
        if ((time[i] < time[j] && risk_score[i] > risk_score[j]) ||
            (time[j] < time[i] && risk_score[j] > risk_score[i])) {
          concord <- concord + 1
        } else if (risk_score[i] == risk_score[j]) {
          par_concord <- par_concord + 0.5
        }
      } else {
        if (status[i] + status[j] > 0) {
          if (risk_score[i] == risk_score[j]) {
            concord <- concord + 1
          } else {
            par_concord <- par_concord + 0.5
          }
        }
      }
    }
  }

  c_index <- (concord + par_concord) / permissible
  names(c_index) <- "c-index"
  return(round(c_index, 6))
}


#' Brier Score for Right-Censored Survival Data at a Fixed Time
#'
#' Computes the Brier score at a fixed time point using inverse probability of censoring weights (IPCW).
#'
#' @param object A `Surv` object with observed time and status.
#' @param pre_sp A numeric vector of predicted survival probabilities at `t_star`.
#' @param t_star The evaluation time point.
#'
#' @return A single numeric value representing the Brier score.
#' @export
#'
#' @examples
#' \donttest{
#' library(survival)
#' data(veteran, package = "survival")
#' mod <- survdnn(Surv(time, status) ~
#' age + karno + celltype, data = veteran, epochs = 50, verbose = FALSE)
#' pred <- predict(mod, newdata = veteran, type = "survival", times = c(30, 90, 180))
#' y <- model.response(model.frame(mod$formula, veteran))
#' survdnn::brier(y, pre_sp = pred[["t=90"]], t_star = 90)
#' }

brier <- function(object, pre_sp, t_star) {
  if (!inherits(object, "Surv")) stop("object must be a survival object")
  if (length(pre_sp) != nrow(object)) stop("Length of predictions must match number of observations")

  time <- object[, 1]
  status <- object[, 2]
  km_fit <- survfit(Surv(time, 1 - status) ~ 1)

  all_times <- sort(unique(c(t_star, time)))
  G_all <- summary(km_fit, times = all_times, extend = TRUE)$surv
  names(G_all) <- as.character(all_times)

  Gt <- G_all[as.character(t_star)]

  if (is.na(Gt) || Gt == 0) {
    warning("brier: G(t_star) is NA or zero at t = ", t_star, "; returning NA.")
    return(NA_real_)
  }

  early_event <- which(time < t_star & status == 1)
  later_risk  <- which(time >= t_star)
  score_vec <- numeric(length(time))

  if (length(early_event) > 0) {
    Gti <- G_all[as.character(time[early_event])]
    valid <- which(!is.na(Gti) & Gti > 0)
    score_vec[early_event[valid]] <- (pre_sp[early_event[valid]]^2) / Gti[valid]
  }

  if (length(later_risk) > 0) {
    score_vec[later_risk] <- ((1 - pre_sp[later_risk])^2) / Gt
  }

  bs_value <- mean(score_vec, na.rm = TRUE)
  names(bs_value) <- "brier"
  return(round(bs_value, 6))
}




#' Integrated Brier Score (IBS) from a Survival Probability Matrix
#'
#' Computes the Integrated Brier Score (IBS) over a set of evaluation time points,
#' using trapezoidal integration and IPCW adjustment for right-censoring.
#'
#' @param object A `Surv` object with observed time and status.
#' @param sp_matrix A data frame or matrix of predicted survival probabilities.
#'   Each column corresponds to a time point in `times`.
#' @param times A numeric vector of time points. Must match the columns of `sp_matrix`.
#'
#' @return A single numeric value representing the integrated Brier score.
#' @export
#'
#' @examples
#' \donttest{
#' set.seed(123)
#' library(survival)
#' data(veteran, package = "survival")
#' idx <- sample(nrow(veteran), 0.7 * nrow(veteran))
#' train <- veteran[idx, ]; test <- veteran[-idx, ]
#' mod <- survdnn(Surv(time, status) ~
#' age + karno + celltype, data = train, epochs = 50, verbose = FALSE)
#' pred <- predict(mod, newdata = test, times = c(30, 90, 180), type = "survival")
#' y_test <- model.response(model.frame(mod$formula, test))
#' ibs_survmat(y_test, sp_matrix = pred, times = c(30, 90, 180))
#' }

ibs_survmat <- function(object, sp_matrix, times) {
  if (!inherits(object, "Surv")) stop("object must be a survival object")
  if (length(times) != ncol(sp_matrix)) stop("Length of times must match sp_matrix columns")

  if (length(times) == 1) {
    return(brier(object, pre_sp = sp_matrix[, 1], t_star = times[1]))
  }

  times <- sort(times)
  brier_vec <- vapply(seq_along(times), function(i) {
    brier(object, pre_sp = sp_matrix[, i], t_star = times[i])
  }, numeric(1))

  dt <- diff(times)
  integral <- sum(brier_vec[-length(brier_vec)] * dt)
  ibs_value <- integral / (max(times) - min(times))
  names(ibs_value) <- "ibs"
  return(round(ibs_value, 6))
}
