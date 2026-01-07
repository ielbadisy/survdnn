#' Summarize a Deep Survival Neural Network Model
#'
#' Provides a structured summary of a fitted `survdnn` model, including the network architecture,
#' training configuration, and data characteristics. The summary is printed automatically with
#' a styled header and sectioned output using \{cli\} and base formatting. The object is returned invisibly.
#'
#' @param object An object of class `"survdnn"` returned by the [survdnn()] function.
#' @param ... Currently ignored (for future compatibility).
#'
#' @return Invisibly returns an object of class `"summary.survdnn"`.
#' @export
#'
#' @examples
#' \donttest{
#' set.seed(42)
#' sim_data <- data.frame(
#'   age = rnorm(100, 60, 10),
#'   sex = factor(sample(c("male", "female"), 100, TRUE)),
#'   trt = factor(sample(c("A", "B"), 100, TRUE)),
#'   time = rexp(100, 0.05),
#'   status = rbinom(100, 1, 0.7)
#' )
#' mod <- survdnn(Surv(time, status) ~ age + sex + trt, data = sim_data, epochs = 50, verbose = FALSE)
#' summary(mod)
#' }

summary.survdnn <- function(object, ...) {
  stopifnot(inherits(object, "survdnn"))

  # Rebuild model frame using the same NA policy as during fitting
  na_action <- if (!is.null(object$na_action)) object$na_action else "omit"
  mf <- model.frame(
    object$formula,
    data = object$data,
    na.action = if (na_action == "omit") stats::na.omit else stats::na.fail
  )

  y <- model.response(mf)
  time <- y[, "time"]
  status <- y[, "status"]

  n_before <- nrow(object$data)
  n_used   <- nrow(mf)
  n_removed <- n_before - n_used

  events   <- sum(status == 1, na.rm = TRUE)
  censored <- sum(status == 0, na.rm = TRUE)

  out <- list(
    model_summary = list(
      hidden_layers = object$hidden,
      activation    = object$activation,
      dropout       = if (!is.null(object$dropout)) object$dropout else NA_real_,
      batch_norm    = if (!is.null(object$batch_norm)) object$batch_norm else NA,
      final_loss    = object$final_loss,
      loss_history_length = if (!is.null(object$loss_history)) length(object$loss_history) else NA_integer_
    ),
    training_summary = list(
      epochs        = object$epochs,
      learning_rate = object$lr,
      loss_function = object$loss,
      optimizer     = object$optimizer,
      device        = object$device,
      na_action     = object$na_action
    ),
    data_summary = list(
      observations_total = n_before,
      observations_used  = n_used,
      observations_removed = n_removed,
      predictors   = object$xnames,
      n_predictors = length(object$xnames),
      time_range   = range(time, na.rm = TRUE),
      events       = events,
      censored     = censored,
      event_rate   = mean(status == 1, na.rm = TRUE),
      standardized = !is.null(object$x_center) && !is.null(object$x_scale)
    ),
    loss_details = NULL,
    formula = object$formula
  )

  if (identical(object$loss, "aft")) {
    out$loss_details <- list(
      aft_loc       = object$aft_loc,
      aft_log_sigma = object$aft_log_sigma
    )
  } else if (identical(object$loss, "coxtime")) {
    out$loss_details <- list(
      coxtime_time_center = object$coxtime_time_center,
      coxtime_time_scale  = object$coxtime_time_scale
    )
  }

  class(out) <- "summary.survdnn"

  cli::cli_h1("Summary of survdnn model")

  cat("\nFormula:\n  ")
  print(out$formula)

  cat("\nModel architecture:\n")
  cat("  Hidden layers: ", paste(out$model_summary$hidden_layers, collapse = " : "), "\n")
  cat("  Activation: ", out$model_summary$activation, "\n")
  cat("  Dropout: ", out$model_summary$dropout, "\n")
  cat("  Batch norm: ", out$model_summary$batch_norm, "\n")
  cat("  Final loss: ", formatC(out$model_summary$final_loss, digits = 6, format = "f"), "\n")

  cat("\nTraining summary:\n")
  cat("  Epochs: ", out$training_summary$epochs, "\n")
  cat("  Learning rate: ", out$training_summary$learning_rate, "\n")
  cat("  Loss function: ", out$training_summary$loss_function, "\n")
  cat("  Optimizer: ", out$training_summary$optimizer, "\n")
  cat("  Device: ", as.character(out$training_summary$device), "\n")
  cat("  NA action: ", as.character(out$training_summary$na_action), "\n")

  cat("\nData summary:\n")
  cat("  Observations (used/total): ", out$data_summary$observations_used, " / ",
      out$data_summary$observations_total, "\n", sep = "")
  if (out$data_summary$observations_removed > 0) {
    cat("  Removed due to missing values: ", out$data_summary$observations_removed, "\n", sep = "")
  }
  cat("  Predictors (", out$data_summary$n_predictors, "): ",
      paste(out$data_summary$predictors, collapse = ", "), "\n", sep = "")
  cat("  Time range: [", paste(out$data_summary$time_range, collapse = ", "), "]\n")
  cat("  Events / censored: ", out$data_summary$events, " / ", out$data_summary$censored, "\n")
  cat("  Event rate: ", sprintf("%.1f%%", 100 * out$data_summary$event_rate), "\n")
  cat("  Predictors standardized: ", if (isTRUE(out$data_summary$standardized)) "yes" else "no", "\n")

  if (!is.null(out$loss_details)) {
    cat("\nLoss-specific details:\n")
    for (nm in names(out$loss_details)) {
      cat("  ", nm, ": ", out$loss_details[[nm]], "\n", sep = "")
    }
  }

  invisible(out)
}
