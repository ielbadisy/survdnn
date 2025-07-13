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

  if (requireNamespace("torch", quietly = TRUE)) {
    if (!torch::torch_is_installed()) {
      try(torch::install_torch(), silent = TRUE)
    }
    try(torch::torch_tensor(0), silent = TRUE)
  }
}

## handles user-facing messaging
.onAttach <- function(libname, pkgname) {
  if (!requireNamespace("torch", quietly = TRUE)) return()

  if (!torch::torch_is_installed() && interactive()) {
    msg <- paste0(
      cli::rule("Torch Backend Not Installed", line_col = "red"),
      "\nTorch will be automatically installed when needed.",
      "\nIf installation fails, you can run manually: ", cli::col_yellow("install_torch()"),
      "\nDocs: https://torch.mlverse.org/docs/articles/installation.html"
    )
    packageStartupMessage(msg)
  }
}
