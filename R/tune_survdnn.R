#' Tune survdnn model hyperparameters (explicit loss support)
#' @export
tune_survdnn <- function(formula, data, times, metrics = "cindex",
                         param_grid, folds = 3, seed = 42,
                         refit = FALSE,
                         return = c("all", "summary", "best_model")) {

  return <- match.arg(return)
  param_df <- tidyr::crossing(!!!param_grid)

  all_results <- purrr::pmap_dfr(param_df, function(hidden, lr, activation, epochs, .loss_fn, loss_name) {
    config_tbl <- tibble::tibble(
      hidden = list(hidden),
      lr = lr,
      activation = activation,
      epochs = epochs,
      .loss_fn = list(.loss_fn),
      loss_name = loss_name
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

  # Select best configuration
  primary_metric <- metrics[1]
  best_row <- summary_tbl |>
    dplyr::filter(metric == primary_metric) |>
    dplyr::slice_max(order_by = if (primary_metric %in% c("cindex")) mean else -mean, n = 1)

  best_config <- best_row |>
    dplyr::select(any_of(c("hidden", "lr", "activation", "epochs", ".loss_fn")))

  if (refit) {
    message("Refitting best model on full data...")
    best_model <- do.call(survdnn, c(list(formula = formula, data = data), best_config))
  }

  switch(return,
         "all" = all_results,
         "summary" = summary_tbl,
         "best_model" = if (refit) best_model else stop("refit must be TRUE to return best_model.")
  )
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




grid <- list(
  hidden     = list(c(16), c(32, 16)),
  lr         = c(1e-4),
  activation = c("relu"),
  epochs     = c(300),
  .loss_fn   = list(cox_loss, aft_loss),
  loss_name  = c("cox_loss", "aft_loss")
)



library(survival)
data(veteran)

# Run tuning
tune_res <- tune_survdnn(
  formula = Surv(time, status) ~ age + karno + celltype,
  data = veteran,
  times = c(90),
  metrics = c("cindex"),
  param_grid = grid,
  folds = 3,
  seed = 42,
  refit = FALSE,
  return = "all"
)

# Summarize
summary_tbl <- summarize_tune_survdnn(tune_res)

summary_tbl
