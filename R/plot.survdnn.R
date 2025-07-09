
plot.survdnn <- function(x, newdata = NULL, times = 1:365,
                         add_mean = TRUE, mean_col = "black",
                         mean_lty = 2, mean_lwd = 3, ...) {
  stopifnot(inherits(x, "survdnn"))

  if (is.null(newdata)) {
    newdata <- x$data[1:5, , drop = FALSE]
  }

  survmat <- predict(x, newdata = newdata, times = times, type = "survival")
  n_obs <- nrow(survmat)

  # generate a color palette excluding black
  curve_colors <- grDevices::rainbow(n_obs)
  curve_colors[curve_colors == mean_col] <- "#999999"  # fallback to gray if collision

  # plot individual survival curves
  matplot(times, t(as.matrix(survmat)), type = "l", lty = 1,
          col = curve_colors, lwd = 2,
          xlab = "Time", ylab = "Survival Probability",
          main = "Predicted Survival Curves")

  # add mean survival curve in black
  if (add_mean) {
    mean_surv <- colMeans(survmat, na.rm = TRUE)
    lines(times, mean_surv, col = mean_col, lty = mean_lty, lwd = mean_lwd)
  }

  invisible(survmat)
}

library(survival)
library(ggplot2)

data(veteran)

mod <- survdnn(Surv(time, status) ~ age + karno + celltype, data = veteran,
               hidden = c(32, 16), activation = "relu", epochs = 300, verbose = FALSE)


plot(mod, times = 1:365)
plot(mod, newdata = veteran[1:10, ])
