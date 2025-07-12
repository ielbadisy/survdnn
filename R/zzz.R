.onLoad <- function(libname, pkgname) {
  
  # package level default options
  op <- options()
  op.survdnn <- list(survdnn.default_epochs = 100)
  toset <- !(names(op.survdnn) %in% names(op))
  if (any(toset)) options(op.survdnn[toset])

  # declare global variables to avoid R CMD check notes
  utils::globalVariables(c(
    "fold", "metric", "value", "id", "time", "surv", "group", "mean_surv",
    "n", "se", "hidden", "lr", "activation", "epochs", "loss_name", ".loss_fn"
  ))
}
