cindex_survmat <- function(object, predicted, t_star = NULL) {
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
  n <- length(time)

  concord <- 0
  par_concord <- 0
  permissible <- 0

  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      if ((time[i] < time[j] && status[i] == 0) || 
          (time[j] < time[i] && status[j] == 0)) next
      if (time[i] == time[j] && status[i] == 0 && status[j] == 0) next

      permissible <- permissible + 1

      if (time[i] != time[j]) {
        if ((time[i] < time[j] && risk_score[i] > risk_score[j]) || 
            (time[j] < time[i] && risk_score[j] > risk_score[i])) {
          concord <- concord + 1
        } else if (risk_score[i] == risk_score[j]) {
          par_concord <- par_concord + 0.5
        }
      } else {
        if (status[i] + status[j] > 0) {
          if (risk_score[i] == risk_score[j]) {
            concord <- concord + 1
          } else {
            par_concord <- par_concord + 0.5
          }
        }
      }
    }
  }

  c_index <- (concord + par_concord) / permissible
  names(c_index) <- "c-index"
  return(round(c_index, 6))
}

brier <- function(object, pre_sp, t_star) {
  if (!inherits(object, "Surv")) stop("object must be a survival object")
  if (length(pre_sp) != nrow(object)) stop("Length of predictions must match number of observations")

  time <- object[, 1]
  status <- object[, 2]
  km_fit <- survfit(Surv(time, 1 - status) ~ 1)

  all_times <- sort(unique(c(t_star, time)))
  G_all <- summary(km_fit, times = all_times, extend = TRUE)$surv
  names(G_all) <- as.character(all_times)

  Gt <- G_all[as.character(t_star)]

  if (is.na(Gt) || Gt == 0) {
    warning("brier: G(t_star) is NA or zero at t = ", t_star, "; returning NA.")
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

  if (length(later_risk) > 0) {
    score_vec[later_risk] <- ((1 - pre_sp[later_risk])^2) / Gt
  }

  bs_value <- mean(score_vec, na.rm = TRUE)
  names(bs_value) <- "brier"
  return(round(bs_value, 6))
}



brier_ibs_survmat <- function(object, sp_matrix, times) {
  if (!inherits(object, "Surv")) stop("object must be a survival object")
  if (length(times) != ncol(sp_matrix)) stop("Length of times must match sp_matrix columns")

  if (length(times) == 1) {
    return(brier(object, pre_sp = sp_matrix[, 1], t_star = times[1]))
  }

  times <- sort(times)
  brier_vec <- vapply(seq_along(times), function(i) {
    brier(object, pre_sp = sp_matrix[, i], t_star = times[i])
  }, numeric(1))

  dt <- diff(times)
  integral <- sum(brier_vec[-length(brier_vec)] * dt)
  ibs_value <- integral / (max(times) - min(times))
  names(ibs_value) <- "ibs"
  return(round(ibs_value, 6))
}


#-------TEST 

library(survival)
library(torch)

data(veteran)
set.seed(42)
train_idx <- sample(nrow(veteran), 0.7 * nrow(veteran))
train_data <- veteran[train_idx, ]
test_data  <- veteran[-train_idx, ]

# Fit survdnn model
mod <- survdnn(Surv(time, status) ~ age + karno + celltype,
               data = train_data,
               hidden = c(32, 16),
               epochs = 100,
               verbose = FALSE)

# prediction time
eval_times <- c(30, 90, 180)

# compute predicted survival probas
pred_surv <- predict(mod, newdata = test_data, times = eval_times, type = "survival")

# extract real survival outcome
y_test <- model.response(model.frame(mod$formula, test_data))

## C-index
cindex_survmat(y_test, predicted = pred_surv, t_star = max(eval_times))

## brier score in a single time
brier(y_test, pre_sp = pred_surv[["t=90"]], t_star = 90)

# ibs 
brier_ibs_survmat(y_test, pred_surv, eval_times)
