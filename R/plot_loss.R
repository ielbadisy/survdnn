utils::globalVariables(c("epoch"))

#' Plot Training Loss for a survdnn Model
#'
#' Visualize the evolution of the training loss across epochs for a fitted
#' `survdnn` model. Helps inspect convergence, instability, or callback effects
#' (e.g., early stopping).
#'
#' @param object A fitted `survdnn` model.
#' @param smooth Logical; if `TRUE`, overlays a smoothed loess curve.
#' @param log_y Logical; if `TRUE`, uses a log10 y-scale.
#' @param ... Reserved for future use.
#'
#' @return A `ggplot` object.
#' @export
plot_loss <- function(object,
                      smooth = FALSE,
                      log_y  = FALSE,
                      ...) {
  stopifnot(inherits(object, "survdnn"))

  loss_history <- object$loss_history
  if (is.null(loss_history) || length(loss_history) == 0) {
    stop("Object has no `loss_history` to plot.", call. = FALSE)
  }

  df <- data.frame(
    epoch = seq_along(loss_history),
    loss  = as.numeric(loss_history)
  )

  p <- ggplot2::ggplot(df, ggplot2::aes(x = epoch, y = loss)) +
    ggplot2::geom_line() +
    ggplot2::labs(
      title = "survdnn training loss",
      x     = "Epoch",
      y     = "Loss"
    ) +
    ggplot2::theme_minimal()

  if (isTRUE(smooth))
    p <- p + ggplot2::geom_smooth(se = FALSE, method = "loess")

  if (isTRUE(log_y))
    p <- p + ggplot2::scale_y_log10()

  p
}
