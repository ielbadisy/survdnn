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

evaluate_survlearner <- function(model,
                                 metrics = c("cindex", "ibs", "iae", "ise", "brier"),
                                 times) {
  # Check inputs
  if (missing(model) || !is.list(model)) stop("Must provide a fitted model object.")
  if (missing(times)) stop("Must provide time points for evaluation.")

  # Extract stored components
  formula <- model$formula
  data    <- model$data
  engine  <- attr(model, "engine")

  if (is.null(engine)) stop("Model object must have an 'engine' attribute.")

  # Infer prediction function
  pred_fun_name <- paste0("predict_", engine)
  if (!exists(pred_fun_name, mode = "function")) {
    stop("Prediction function not found: ", pred_fun_name)
  }

  pred_fun <- get(pred_fun_name, mode = "function")

  # Run prediction on training data
  sp_matrix <- pred_fun(model, newdata = data, times = times)

  # Extract outcome: Surv(time, status) or Surv(time, status == event)
  tf <- terms(formula, data = data)
  outcome <- attr(tf, "variables")[[2]]
  time_col <- as.character(outcome[[2]])
  status_expr <- outcome[[3]]

  if (is.call(status_expr) && status_expr[[1]] == as.name("==")) {
    status_col <- as.character(status_expr[[2]])
    event_value <- eval(status_expr[[3]], data)
    status_vector <- as.integer(data[[status_col]] == event_value)
  } else {
    status_col <- as.character(status_expr)
    event_value <- 1
    status_vector <- data[[status_col]]
  }

  surv_obj <- survival::Surv(time = data[[time_col]], event = status_vector)

  # Compute metrics
  tibble::tibble(metric = metrics) |>
    dplyr::mutate(value = purrr::map(metric, function(metric) {
      switch(metric,
             "cindex" = Cindex_survmat(surv_obj, predicted = sp_matrix, t_star = max(times)),
             "brier"  = {
               if (length(times) != 1) stop("Brier requires a single time point.")
               Brier(surv_obj, pre_sp = sp_matrix[, 1], t_star = times)
             },
             "ibs"    = Brier_IBS_survmat(surv_obj, sp_matrix, times),
             "iae"    = IAEISE_survmat(surv_obj, sp_matrix, times)["IAE"],
             "ise"    = IAEISE_survmat(surv_obj, sp_matrix, times)["ISE"],
             stop("Unknown metric: ", metric)
      )
    })) |>
    tidyr::unnest(cols = value)
}



