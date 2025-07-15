#' Print a survdnn Model
#'
#' Pretty prints a fitted `survdnn` model. Displays the formula,
#' network architecture, training configuration, and final training loss.
#'
#' @param x An object of class `"survdnn"`, returned by [survdnn()].
#' @param ... Ignored (for future compatibility).
#'
#' @return The model object, invisibly.
#' @export
#'
#' @examples
#' library(survival)
#' data(veteran, package = "survival")
#' mod <- survdnn(Surv(time, status) ~
#' age + karno + celltype, data = veteran, epochs = 20, verbose = FALSE)
#' print(mod)
print.survdnn <- function(x, ...) {
  stopifnot(inherits(x, "survdnn"))

  cli::cli_h1("survdnn model")
  cli::cli_text("Formula: {.val {deparse(x$formula)}}")
  cli::cli_text("Hidden layers: {.val {paste(x$hidden, collapse = ' : ')}}")
  cli::cli_text("Activation: {.val {x$activation}}")
  cli::cli_text("Learning rate: {.val {x$lr}}")
  cli::cli_text("Loss function: {.val {x$loss}}")

  # safely print final loss if it is numeric
  if (is.numeric(x$final_loss) && length(x$final_loss) == 1 && !is.na(x$final_loss)) {
    cli::cli_text("Final loss: {.val {round(x$final_loss, 4)}} after {.val {x$epochs}} epochs")
  } else {
    cli::cli_alert_warning("Final loss is not available.")
  }

  invisible(x)
}
