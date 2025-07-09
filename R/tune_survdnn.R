
tune_survdnn <- function(formula, data, times, metrics = "cindex",
                         param_grid, folds = 3, seed = 42,
                         refit = FALSE,
                         return = c("all", "summary", "best_model")) {

  return <- match.arg(return)

  param_df <- tidyr::crossing(!!!param_grid)

  all_results <- purrr::pmap_dfr(param_df, function(hidden, lr, activation, epochs) {
    config_tbl <- tibble::tibble(
      hidden = list(hidden), lr = lr, activation = activation, epochs = epochs
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
      epochs = epochs
    )

    dplyr::bind_cols(config_tbl[rep(1, nrow(cv_tbl)), ], cv_tbl)
  })

  summary_tbl <- summarize_cv_survdnn(all_results, by_time = FALSE)

  # choose best config based on first metric (e.g., cindex max or ibs min)
  primary_metric <- metrics[1]
  best_row <- summary_tbl |>
    dplyr::filter(metric == primary_metric) |>
    dplyr::slice_max(order_by = if (primary_metric %in% c("cindex")) mean else -mean, n = 1)

  best_config <- best_row |> dplyr::select(any_of(c("hidden", "lr", "activation", "epochs")))

  if (refit) {
    message("Refitting best model on full data...")
    best_model <- do.call(survdnn, c(list(formula = formula, data = data), best_config))
  }

  # return based on user choice
  out <- switch(return,
                "all" = all_results,
                "summary" = summary_tbl,
                "best_model" = if (refit) best_model else stop("refit must be TRUE to return best_model.")
  )
  return(out)
}

# grid definition
grid <- list(
  hidden = list(c(16), c(32, 16)),
  lr = c(1e-4, 5e-4),
  activation = c("relu", "tanh"),
  epochs = c(300)
)

# run tuning + return full results
res_all <- tune_survdnn(
  formula = Surv(time, status) ~ age + karno + celltype,
  data = veteran,
  times = c(90),
  metrics = c("cindex", "ibs"),
  param_grid = grid,
  folds = 3,
  seed = 42,
  refit = FALSE,
  return = "all"
)

# run tuning + return best model fitted on full data
best_mod <- tune_survdnn(
  formula = Surv(time, status) ~ age + karno + celltype,
  data = veteran,
  times = c(90),
  metrics = c("cindex", "ibs"),
  param_grid = grid,
  folds = 3,
  seed = 42,
  refit = TRUE,
  return = "best_model"
)



summarize_tune_survdnn <- function(tuning_results, by_time = TRUE) {
  if (!all(c("metric", "value") %in% names(tuning_results))) {
    stop("Input must be the result of `tune_survdnn(return = 'all')`.")
  }

  group_vars <- c("hidden", "lr", "activation", "epochs", "metric")
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


tune_res <- tune_survdnn(
  formula = Surv(time, status) ~ age + karno + celltype,
  data = veteran,
  times = 1:180,
  metrics = c("cindex", "ibs"),
  param_grid = grid,
  folds = 3,
  seed = 42,
  refit = FALSE,
  return = "all"
)

# summarize all tuning runs
summary_tbl <- summarize_tune_survdnn(tune_res)

# summarize time-resolved metrics (brier)
summary_brier <- summarize_tune_survdnn(tune_res, by_time = TRUE)
