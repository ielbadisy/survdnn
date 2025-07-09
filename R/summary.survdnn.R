
summary.survdnn <- function(object, ...) {
  stopifnot(inherits(object, "survdnn"))

  xmat <- model.matrix(delete.response(terms(object$formula)), object$data)[, object$xnames, drop = FALSE]
  time <- model.response(model.frame(object$formula, object$data))[, "time"]
  status <- model.response(model.frame(object$formula, object$data))[, "status"]

  structure(list(
    model_summary = list(
      hidden_layers = object$hidden,
      activation = object$activation,
      dropout = 0.3,  # hardcoded from build_dnn
      final_loss = object$loss
    ),
    training_summary = list(
      epochs = object$epochs,
      learning_rate = object$lr,
      loss_function = deparse(substitute(object$.loss_fn))
    ),
    data_summary = list(
      observations = nrow(object$data),
      predictors = object$xnames,
      time_range = range(time),
      event_rate = mean(status)
    ),
    formula = object$formula
  ), class = "summary.survdnn")
}



#' Print summary for survdnn
#' @param x An object of class `"summary.survdnn"`.
#' @param ... Ignored.
#' @return Invisibly returns the summary.
#' @export
print.summary.survdnn <- function(x, ...) {
  cat("== Summary of survdnn model ==\n")
  cat("\nFormula:\n  "); print(x$formula)

  cat("\nModel architecture:\n")
  cat("• Hidden layers: ", paste(x$model_summary$hidden_layers, collapse = " → "), "\n")
  cat("• Activation: ", x$model_summary$activation, "\n")
  cat("• Dropout: ", x$model_summary$dropout, "\n")

  cat("\nTraining summary:\n")
  cat("• Epochs: ", x$training_summary$epochs, "\n")
  cat("• Learning rate: ", x$training_summary$learning_rate, "\n")
  cat("• Loss function: ", x$training_summary$loss_function, "\n")
  cat("• Final loss: ", formatC(x$model_summary$final_loss, digits = 6, format = "f"), "\n")

  cat("\nData summary:\n")
  cat("• Observations: ", x$data_summary$observations, "\n")
  cat("• Predictors: ", paste(x$data_summary$predictors, collapse = ", "), "\n")
  cat("• Time range: [", paste(x$data_summary$time_range, collapse = ", "), "]\n")
  cat("• Event rate: ", sprintf("%.1f%%", 100 * x$data_summary$event_rate), "\n")

  invisible(x)
}



mod <- survdnn(Surv(time, status) ~ age + karno + celltype, data = veteran)
summary(mod)
