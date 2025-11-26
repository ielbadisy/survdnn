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

  mf <- model.frame(object$formula, object$data)
  xmat <- model.matrix(delete.response(terms(object$formula)), mf)[, object$xnames, drop = FALSE]
  y <- model.response(mf)
  time <- y[, "time"]
  status <- y[, "status"]

  out <- list(
    model_summary = list(
      hidden_layers = object$hidden,
      activation = object$activation,
      dropout = if (!is.null(object$dropout)) object$dropout else 0.3,
      final_loss = object$final_loss
    ),
    training_summary = list(
      epochs        = object$epochs,
      learning_rate = object$lr,
      loss_function = object$loss,
      optimizer     = if (!is.null(object$optimizer)) object$optimizer else "adam"
    ),
    data_summary = list(
      observations = nrow(object$data),
      predictors   = object$xnames,
      time_range   = range(time),
      event_rate   = mean(status)
    ),
    formula = object$formula
  )
  class(out) <- "summary.survdnn"

  cli::cli_h1("Summary of survdnn model")

  cat("\nFormula:\n  ")
  print(out$formula)

  cat("\nModel architecture:\n")
  cat("  Hidden layers: ", paste(out$model_summary$hidden_layers, collapse = " : "), "\n")
  cat("  Activation: ", out$model_summary$activation, "\n")
  cat("  Dropout: ", out$model_summary$dropout, "\n")
  cat("  Final loss: ", formatC(out$model_summary$final_loss, digits = 6, format = "f"), "\n")

  cat("\nTraining summary:\n")
  cat("  Epochs: ", out$training_summary$epochs, "\n")
  cat("  Learning rate: ", out$training_summary$learning_rate, "\n")
  cat("  Loss function: ", out$training_summary$loss_function, "\n")
  cat("  Optimizer: ", out$training_summary$optimizer, "\n")

  cat("\nData summary:\n")
  cat("  Observations: ", out$data_summary$observations, "\n")
  cat("  Predictors: ", paste(out$data_summary$predictors, collapse = ", "), "\n")
  cat("  Time range: [", paste(out$data_summary$time_range, collapse = ", "), "]\n")
  cat("  Event rate: ", sprintf("%.1f%%", 100 * out$data_summary$event_rate), "\n")

  invisible(out)
}
