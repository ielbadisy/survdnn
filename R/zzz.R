## handles silent setup
.onLoad <- function(libname, pkgname) {
  op <- options()
  op.survdnn <- list(survdnn.default_epochs = 100)
  toset <- !(names(op.survdnn) %in% names(op))
  if (any(toset)) options(op.survdnn[toset])

  utils::globalVariables(c(
    "fold", "metric", "value", "id", "time", "surv", "group", "mean_surv",
    "n", "se", "hidden", "lr", "activation", "epochs", "loss_name", ".loss_fn"
  ))

  # IMPORTANT: never load or probe torch here (because CRAN/Windows may segfault).
  # No torch checks, no tensor creation on load.
}

## handles user-facing messaging
.onAttach <- function(libname, pkgname) {
  # Do NOT load torch or call torch::torch_is_installed() here.
  # friendly hint that doesn't load the namespace:
  torch_pkg_present <- nzchar(system.file(package = "torch"))

  if (interactive() && !torch_pkg_present) {
    packageStartupMessage(
      "Optional dependency 'torch' not found. ",
      "Install the R package 'torch' and then run torch::install_torch() ",
      "to use deep-learning features."
    )
  }
  invisible()
}
