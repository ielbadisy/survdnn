#' Plot survdnn Survival Curves using ggplot2
#'
#' Visualizes survival curves predicted by a fitted `survdnn` model.
#' Curves can be grouped by a categorical variable in `newdata` and
#' optionally display only the group-wise means or overlay them.
#'
#' @param x A fitted `survdnn` model object.
#' @param newdata Optional data frame for prediction (defaults to training data).
#' @param times A numeric vector of time points at which to compute survival probabilities.
#' @param group_by Optional name of a column in `newdata` used to color and group curves.
#' @param plot_mean_only Logical; if `TRUE`, plots only the mean survival curve per group.
#' @param add_mean Logical; if `TRUE`, adds mean curves to the individual lines.
#' @param alpha Alpha transparency for individual curves (ignored if `plot_mean_only = TRUE`).
#' @param mean_lwd Line width for mean survival curves.
#' @param mean_lty Line type for mean survival curves.
#' @param ... Reserved for future use.
#'
#' @return A `ggplot` object.
#' @export
#'
#' @examples
#' \donttest{
#' library(survival)
#' data(veteran)
#' set.seed(42)
#' \donttest{
#' mod <- survdnn(Surv(time, status) ~ age + karno + celltype, data = veteran,
#'                hidden = c(16, 8), epochs = 100, verbose = FALSE)
#' plot(mod, group_by = "celltype", times = 1:300)
#' }
#' }
plot.survdnn <- function(x, newdata = NULL, times = 1:365,
                         group_by = NULL,
                         plot_mean_only = FALSE,
                         add_mean = TRUE,
                         alpha = 0.3,
                         mean_lwd = 1.3,
                         mean_lty = 1,
                         ...) {
  stopifnot(inherits(x, "survdnn"))
  if (is.null(newdata)) newdata <- x$data

  # compute survival probabilities
  survmat <- predict(x, newdata = newdata, times = times, type = "survival")
  df_surv <- as.data.frame(survmat)
  df_surv$id <- seq_len(nrow(df_surv))

  # reshape to long format
  df_long <- tidyr::pivot_longer(
    df_surv,
    cols = -id,
    names_to = "time_label",
    values_to = "surv"
  )

  # clean up time labels
  df_long$time <- as.numeric(gsub("t=", "", df_long$time_label))

  # group handling
  if (!is.null(group_by)) {
    if (!group_by %in% names(newdata)) {
      stop("Column '", group_by, "' not found in newdata.")
    }
    df_long$group <- rep(newdata[[group_by]], each = length(times))
  } else {
    df_long$group <- "all"
  }

  # base ggplot
  p <- ggplot(df_long, aes(x = time, y = surv, group = id, color = group))

  # plot individual curves
  if (!plot_mean_only) {
    p <- p + geom_line(alpha = alpha, linewidth = 0.7)
  }

  # plot mean curves
  if (add_mean || plot_mean_only) {
    df_mean <- dplyr::group_by(df_long, group, time) |>
      dplyr::summarise(mean_surv = mean(surv, na.rm = TRUE), .groups = "drop")

    p <- p + geom_line(data = df_mean,
                       mapping = aes(x = time, y = mean_surv, color = group),
                       linewidth = mean_lwd,
                       linetype = mean_lty,
                       inherit.aes = FALSE)
  }

  # finalize
  p <- p +
    theme_minimal() +
    labs(
      title = "Predicted Survival Curves",
      x = "Time",
      y = "Survival Probability",
      color = if (!is.null(group_by)) group_by else NULL
    )

  return(p)
}
