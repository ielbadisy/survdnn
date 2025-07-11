evaluate_survdnn <- function(model, metrics = c("cindex", "brier", "ibs"), times, newdata = NULL) {
  stopifnot(inherits(model, "survdnn"))
  if (missing(times)) stop("You must provide `times` for evaluation.")
  
  allowed_metrics <- c("cindex", "brier", "ibs")
  unknown <- setdiff(metrics, allowed_metrics)
  
  if (length(unknown) > 0) stop("Unknown metric(s): ", paste(unknown, collapse = ", "))
  
  data <- if (is.null(newdata)) model$data else newdata
  sp_matrix <- predict(model, newdata = data, times = times, type = "survival")  # ✅ ici: sp_matrix, pas p_matrix
  
  # extract Surv outcome
  mf <- model.frame(model$formula, data)
  y <- model.response(mf)
  
  if (!inherits(y, "Surv")) stop("The outcome must be a 'Surv' object.")
  
  results <- purrr::map_dfr(metrics, function(metric) {
    if (metric == "brier" && length(times) > 1) {
      tibble::tibble(
        metric = "brier",
        time = times,
        value = vapply(seq_along(times), function(i) {
          brier(y, pre_sp = sp_matrix[, i], t_star = times[i])
        }, numeric(1))
      )
    } else {
      val <- switch(metric,
        "cindex" = cindex_survmat(y, predicted = sp_matrix, t_star = max(times)),
        "brier"  = brier(y, pre_sp = sp_matrix[, 1], t_star = times[1]),
        "ibs"    = brier_ibs_survmat(y, sp_matrix, times)
      )
      
      tibble::tibble(metric = metric, value = val)
    }
  })
  
  return(results)
}



#---------- TEST
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

# evaluate_survdnn()
evaluate_survdnn(mod, metrics = c("cindex", "ibs"), times = eval_times, newdata = test_data)
evaluate_survdnn(mod, metrics = c("brier"), times = eval_times, newdata = test_data)
