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
  # never load or probe torch here (because CRAN/Windows may segfault)
}

## handles user-facing messaging
.onAttach <- function(libname, pkgname) {
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


## set R + torch seeds safely
survdnn_set_seed <- function(.seed = NULL) {
  if (is.null(.seed)) return(invisible(NULL))

  # R RNG
  set.seed(.seed)

  # torch RNG (only if available + backend installed)
  if (requireNamespace("torch", quietly = TRUE)) {
    if (isTRUE(torch::torch_is_installed())) {
      torch::torch_manual_seed(.seed)
    }
  }

  invisible(NULL)
}




## internal utility to choose a torch device for survdnn
survdnn_get_device <- function(.device = c("auto", "cpu", "cuda")) {
  .device <- match.arg(.device)

  if (!requireNamespace("torch", quietly = TRUE)) {
    stop(
      "The 'torch' package is required to fit survdnn models.\n",
      "Please install it with: install.packages('torch') and then run torch::install_torch().",
      call. = FALSE
    )
  }

  if (!isTRUE(torch::torch_is_installed())) {
    stop(
      "The Torch backend is not installed.\n",
      "Please run: torch::install_torch().",
      call. = FALSE
    )
  }

  if (.device == "cpu") {
    return(torch::torch_device("cpu"))
  }

  if (.device == "cuda") {
    if (!torch::cuda_is_available()) {
      warning("CUDA was requested but is not available; falling back to CPU.")
      return(torch::torch_device("cpu"))
    }
    return(torch::torch_device("cuda"))
  }

  if (torch::cuda_is_available()) {
    torch::torch_device("cuda")
  } else {
    torch::torch_device("cpu")
  }
}
