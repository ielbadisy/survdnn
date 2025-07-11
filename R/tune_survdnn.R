tune_survdnn <- function(formula, data, times, metrics = "cindex",
                         param_grid, folds = 3, seed = 42,
                         refit = FALSE,
                         return = c("all", "summary", "best_model")) {

  return <- match.arg(return)
  param_df <- tidyr::crossing(!!!param_grid)

  all_results <- purrr::pmap_dfr(param_df, function(hidden, lr, activation, epochs, .loss_fn, loss_name) {
    config_tbl <- tibble::tibble(
      hidden     = list(hidden),
      lr         = lr,
      activation = activation,
      epochs     = epochs,
      .loss_fn   = list(.loss_fn),
      loss_name  = loss_name
    )

    cv_tbl <- cv_survdnn(
      formula = formula,
      data = data,
      times = times,
      metrics = metrics,
      folds = folds,
      seed = seed,
      hidden = hidden,
      lr = lr,
      activation = activation,
      epochs = epochs,
      .loss_fn = .loss_fn
    )

    dplyr::bind_cols(config_tbl[rep(1, nrow(cv_tbl)), ], cv_tbl)
  })

  summary_tbl <- summarize_tune_survdnn(all_results, by_time = FALSE)

  # select best configuration from all_results
  primary_metric <- metrics[1]
  best_row_all <- all_results |>
    dplyr::filter(metric == primary_metric) |>
    dplyr::group_by(hidden, lr, activation, epochs, .loss_fn, loss_name) |>
    dplyr::summarise(mean = mean(value, na.rm = TRUE), .groups = "drop") |>
    dplyr::slice_max(order_by = if (primary_metric == "cindex") mean else -mean, n = 1)

  if (refit) {
    message("Refitting best model on full data...")

    best_model <- survdnn(
      formula    = formula,
      data       = data,
      hidden     = best_row_all$hidden[[1]],
      lr         = best_row_all$lr,
      activation = best_row_all$activation,
      epochs     = best_row_all$epochs,
      .loss_fn   = best_row_all$.loss_fn[[1]]
    )
    best_model$loss_name <- best_row_all$loss_name[[1]]
  }

  switch(return,
         "all" = all_results,
         "summary" = summary_tbl,
         "best_model" = if (refit) best_model else dplyr::select(best_row_all, -mean))
}



summarize_tune_survdnn <- function(tuning_results, by_time = TRUE) {
  if (!all(c("metric", "value") %in% names(tuning_results))) {
    stop("Input must be the result of `tune_survdnn(return = 'all')`.")
  }

  group_vars <- c("hidden", "lr", "activation", "epochs", "loss_name", "metric")
  if (by_time && "time" %in% names(tuning_results)) {
    group_vars <- c(group_vars, "time")
  }

  tuning_results |>
    dplyr::group_by(dplyr::across(all_of(group_vars))) |>
    dplyr::summarise(
      mean = mean(value, na.rm = TRUE),
      sd = sd(value, na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::arrange(metric, dplyr::desc(mean))
}




#----- TEST

library(survival)
library(dplyr)
library(tibble)

data(veteran)

grid <- list(
  hidden     = list(c(16), c(32, 16)),
  lr         = c(1e-4),
  activation = c("relu"),
  epochs     = c(100),
  .loss_fn   = list(cox_loss, aft_loss),
  loss_name  = c("cox_loss", "aft_loss")
)
eval_times <- c(90, 300)


## return all results 
tune_all <- tune_survdnn(
  formula = Surv(time, status) ~ age + karno + celltype,
  data = veteran,
  times = eval_times,
  metrics = c("cindex"),
  param_grid = grid,
  folds = 3,
  seed = 42,
  refit = FALSE,
  return = "all"
)

tune_all

## return = "summary" mean CV per config 

tune_summary <- tune_survdnn(
  formula = Surv(time, status) ~ age + karno + celltype,
  data = veteran,
  times = eval_times,
  metrics = c("cindex"),
  param_grid = grid,
  folds = 3,
  seed = 42,
  refit = FALSE,
  return = "summary"
)

tune_summary



## retrun = "best_model"

## when refit = TRUE tune_survdnn() return a model object , when FALSE is retun best values for hyperparameters 

tune_best <- tune_survdnn(
  formula = Surv(time, status) ~ age + karno + celltype,
  data = veteran,
  times = eval_times,
  metrics = c("cindex"),
  param_grid = grid,
  folds = 3,
  seed = 42,
  refit = FALSE,
  return = "best_model"
)

tune_best

