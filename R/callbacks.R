#' Early stopping callback for survdnn
#'
#' Simple early stopping on a monitored scalar, typically the training loss.
#'
#' @param patience Integer. Number of epochs with no improvement before stopping.
#' @param min_delta Minimum change to qualify as an improvement (default: 0).
#' @param mode Character; "min" (for losses) or "max" (for metrics to maximize).
#' @param verbose Logical; whether to print a message when early stopping is triggered.
#'
#' @return A function of the form `function(epoch, current)` that returns TRUE
#'   if training should stop, FALSE otherwise.
#' @export
callback_early_stopping <- function(patience = 10L,
                                    min_delta = 0,
                                    mode = c("min", "max"),
                                    verbose = FALSE) {
  mode <- match.arg(mode)

  if (!is.numeric(patience) || patience < 1) {
    stop("`patience` must be a positive integer.", call. = FALSE)
  }

  best <- if (mode == "min") Inf else -Inf
  wait <- 0L

  function(epoch, current) {
    if (!is.numeric(current) || length(current) != 1L || is.na(current)) {
      warning("callback_early_stopping: `current` must be a single numeric value; ignoring callback for this epoch.")
      return(FALSE)
    }

    improved <- if (mode == "min") {
      current < (best - min_delta)
    } else {
      current > (best + min_delta)
    }

    if (improved) {
      best <<- current
      wait <<- 0L
      return(FALSE)
    }

    wait <<- wait + 1L
    if (wait >= patience) {
      if (verbose) {
        message(
          "Early stopping at epoch ", epoch,
          " (best = ", signif(best, 6), ", patience = ", patience, ")"
        )
      }
      return(TRUE)
    }

    FALSE
  }
}
