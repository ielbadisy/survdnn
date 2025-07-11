#' Plot survdnn survival curves using ggplot2
#'
#' @param x A fitted survdnn object
#' @param newdata Optional new data (defaults to training data)
#' @param times Time grid to compute survival probabilities
#' @param group_by Optional column in `newdata` to group/color curves
#' @param plot_mean_only If TRUE, plots only mean survival per group
#' @param add_mean If TRUE, overlays group-wise mean curves
#' @param alpha Transparency for individual curves (ignored if mean-only)
#' @param mean_lwd Line width for mean curves
#' @param mean_lty Line type for mean curves
#' @param ... Reserved for future use
#'
#' @return A ggplot object
#' @export
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

# compute survival matrix
survmat <- predict(x, newdata = newdata, times = times, type = "survival")
df_surv <- as.data.frame(survmat)
df_surv$id <- seq_len(nrow(df_surv))

# reshape to long format
df_long <- reshape2::melt(df_surv, id.vars = "id", variable.name = "time_label", value.name = "surv")
df_long$time <- as.numeric(gsub("t=", "", df_long$time_label))

# here we add grouping variable if requested
if (!is.null(group_by)) {
df_long$group <- rep(newdata[[group_by]], times = length(times))
} else {
df_long$group <- "all"
}

library(ggplot2)

# store the base ggplot object
p <- ggplot(df_long, aes(x = time, y = surv, group = id, color = group))

if (!plot_mean_only) {
p <- p + geom_line(alpha = alpha, linewidth = 0.7)
}

if (add_mean || plot_mean_only) {
df_mean <- df_long |>
dplyr::group_by(group, time) |>
dplyr::summarise(mean_surv = mean(surv, na.rm = TRUE), .groups = "drop")

p <- p + geom_line(data = df_mean,
mapping = aes(x = time, y = mean_surv, color = group),
linewidth = mean_lwd,
linetype = mean_lty,
inherit.aes = FALSE)
}

p <- p +
theme_minimal()
labs(title = "Predicted Survival Curves",
x = "Time", y = "Survival Probability",
color = if (!is.null(group_by)) group_by else NULL)

return(p)
}


#--- TEST
plot(mod)  # all 
plot(mod, group_by = "celltype", times = 1:300)  # full curves + mean
plot(mod, group_by = "celltype", times = 1:300, plot_mean_only = TRUE)  # mean only
