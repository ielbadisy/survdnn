Cindex_survmat <- function(object, predicted, t_star = NULL) {
  if (!inherits(object, "Surv")) stop("object must be a survival object (from Surv())")
  time <- object[, 1]
  status <- object[, 2]

  if (!is.null(t_star)) {
    t_name <- paste0("t=", t_star)
    if (!(t_name %in% colnames(predicted))) {
      stop("t_star = ", t_star, " not found in predicted survival matrix.")
    }
    surv_prob <- predicted[[t_name]]
  } else {
    surv_prob <- predicted[[ncol(predicted)]]
  }

  risk_score <- 1 - surv_prob

  permissible <- 0
  concord <- 0
  par_concord <- 0
  n <- length(time)

  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      if ((time[i] < time[j] & status[i] == 0) | (time[j] < time[i] & status[j] == 0)) next
      if (time[i] == time[j] & status[i] == 0 & status[j] == 0) next
      permissible <- permissible + 1
      if (time[i] != time[j]) {
        if ((time[i] < time[j] & risk_score[i] > risk_score[j]) |
            (time[j] < time[i] & risk_score[j] > risk_score[i])) {
          concord <- concord + 1
        } else if (risk_score[i] == risk_score[j]) {
          par_concord <- par_concord + 0.5
        }
      } else {
        if (status[i] + status[j] > 0) {
          if (risk_score[i] == risk_score[j]) concord <- concord + 1
          else par_concord <- par_concord + 0.5
        }
      }
    }
  }

  C_index <- (concord + par_concord) / permissible
  names(C_index) <- "C index"
  return(round(C_index, 6))
}

Brier <- function(object, pre_sp, t_star) {
  if (!inherits(object, "Surv")) stop("object must be a survival object")
  if (length(pre_sp) != nrow(object)) stop("Length of predictions must match number of observations")

  time <- object[, 1]
  status <- object[, 2]
  km_fit <- survfit(Surv(time, 1 - status) ~ 1)

  all_times <- sort(unique(c(t_star, time)))
  G_all <- summary(km_fit, times = all_times, extend = TRUE)$surv
  names(G_all) <- as.character(all_times)

  Gt <- G_all[as.character(t_star)]

  # instead of stopping, return NA and warn if Gt is NA or zero ++++
  if (is.na(Gt) || Gt == 0) {
    warning("Brier: Gt(t_star) is NA or zero at t = ", t_star, "; returning NA.")
    return(NA_real_)
  }

  early_event <- which(time < t_star & status == 1)
  later_risk  <- which(time >= t_star)
  score_vec <- numeric(length(time))

  if (length(early_event) > 0) {
    Gti <- G_all[as.character(time[early_event])]
    valid <- which(!is.na(Gti) & Gti > 0)
    score_vec[early_event[valid]] <- (pre_sp[early_event[valid]]^2) / Gti[valid]
  }

  if (Gt > 0 && length(later_risk) > 0) {
    score_vec[later_risk] <- ((1 - pre_sp[later_risk])^2) / Gt
  }

  BSvalue <- mean(score_vec, na.rm = TRUE)
  names(BSvalue) <- "Brier Score"
  return(round(BSvalue, 6))
}

Brier_IBS_survmat <- function(object, sp_matrix, times) {
  if (!inherits(object, "Surv")) stop("object must be a survival object")
  if (length(times) != ncol(sp_matrix)) stop("Length of times must match sp_matrix columns")
  if (length(times) == 1) {
    return(Brier(object, pre_sp = sp_matrix[, 1], t_star = times[1]))
  }

  times <- sort(times)
  brier_vec <- vapply(seq_along(times), function(i) {
    Brier(object, pre_sp = sp_matrix[, i], t_star = times[i])
  }, numeric(1))

  dt <- diff(times)
  integral <- sum(brier_vec[-length(brier_vec)] * dt)
  IBS_value <- integral / (max(times) - min(times))
  names(IBS_value) <- "IBS"
  return(round(IBS_value, 6))
}

IAEISE_survmat <- function(object, sp_matrix, times) {
  if (!inherits(object, "Surv")) stop("object must be a survival object")
  if (length(times) != ncol(sp_matrix)) stop("Length of times must match sp_matrix columns")

  mean_pred_surv <- colMeans(sp_matrix)
  km_fit <- survfit(object ~ 1)
  km_data <- data.frame(time = km_fit$time, surv = km_fit$surv)
  km_data <- km_data[km_fit$n.event > 0, , drop = FALSE]

  pred_at_km <- approx(x = times, y = mean_pred_surv, xout = km_data$time,
                       method = "constant", rule = 2)$y

  dt <- diff(km_data$time)
  t_IAE <- abs(km_data$surv - pred_at_km)
  t_ISE <- (km_data$surv - pred_at_km)^2

  IAE <- sum(t_IAE[-length(t_IAE)] * dt)
  ISE <- sum(t_ISE[-length(t_ISE)] * dt)
  return(round(c(IAE = IAE, ISE = ISE), 4))
}

evaluate_survdnn <- function(model,
                             metrics = c("cindex", "ibs", "iae", "ise", "brier"),
                             times,
                             newdata = NULL) {
  stopifnot(inherits(model, "survdnn"))
  if (missing(times)) stop("You must provide `times` for evaluation.")

  data <- if (is.null(newdata)) model$data else newdata
  sp_matrix <- predict(model, newdata = data, times = times, type = "survival")

  # extract Surv outcome
  mf <- model.frame(model$formula, data)
  y <- model.response(mf)
  if (!inherits(y, "Surv")) stop("The outcome must be a 'Surv' object.")

  # Mmtric computation
  results <- purrr::map_dfr(metrics, function(metric) {
    if (metric == "brier" && length(times) > 1) {
      tibble::tibble(
        metric = "brier",
        time = times,
        value = vapply(seq_along(times), function(i) {
          Brier(y, pre_sp = sp_matrix[, i], t_star = times[i])
        }, numeric(1))
      )
    } else {
      val <- switch(metric,
                    "cindex" = Cindex_survmat(y, predicted = sp_matrix, t_star = max(times)),
                    "brier"  = Brier(y, pre_sp = sp_matrix[, 1], t_star = times[1]),
                    "ibs"    = Brier_IBS_survmat(y, sp_matrix, times),
                    "iae"    = IAEISE_survmat(y, sp_matrix, times)["IAE"],
                    "ise"    = IAEISE_survmat(y, sp_matrix, times)["ISE"],
                    stop("Unknown metric: ", metric)
      )
      tibble::tibble(metric = metric, value = val)
    }
  })

  return(results)
}

#mod <- survdnn(Surv(time, status) ~ age + karno + celltype, data = veteran)
#evaluate_survdnn(mod, metrics = c("cindex", "ibs"), times = c(30, 90, 180))

library(survival)
data(veteran)
set.seed(42)

# Split into train/test
train_idx <- sample(nrow(veteran), 0.7 * nrow(veteran))
train_data <- veteran[train_idx, ]
test_data  <- veteran[-train_idx, ]

# Fit model on train
mod <- survdnn(Surv(time, status) ~ age + karno + celltype, data = train_data)

# Evaluate on test set using newdata
evaluate_survdnn(mod, metrics = c("cindex", "ibs"), times = c(30, 90, 180), newdata = test_data)

evaluate_survdnn(mod, metrics = "brier", times = c(30, 60, 90), newdata = test_data)




plot_metric_curve <- function(eval_df, metric = "brier") {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("The 'ggplot2' package is required for this plot.")
  }

  df <- eval_df[eval_df$metric == metric & !is.na(eval_df$time), , drop = FALSE]
  if (nrow(df) == 0) {
    stop("No curve data found for metric: ", metric)
  }

  ggplot2::ggplot(df, ggplot2::aes(x = time, y = value)) +
    ggplot2::geom_line(linewidth = 1.1) +
    ggplot2::labs(
      title = paste(toupper(metric), "Score Over Time"),
      x = "Time", y = metric
    ) +
    ggplot2::theme_minimal()
}




eval_brier <- evaluate_survdnn(mod, metrics = "brier", times = 1:365, newdata = test_data)
plot_metric_curve(eval_brier, metric = "brier")

