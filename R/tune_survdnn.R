#' Tune Hyperparameters for a survdnn Model via Cross-Validation
#'
#' Performs k-fold cross-validation over a user-defined hyperparameter grid
#' and selects the best configuration according to the specified evaluation metric.
#'
#' @param formula A survival formula, e.g., `Surv(time, status) ~ x1 + x2`.
#' @param data A data frame.
#' @param times A numeric vector of evaluation time points.
#' @param metrics A character vector of evaluation metrics: "cindex", "brier", or "ibs".
#'               Only the first metric is used for model selection.
#' @param param_grid A named list defining hyperparameter combinations to evaluate,
#'                   passed through [tidyr::crossing()].
#'                   Must include `hidden`, `lr`, `activation`, `epochs`, `.loss_fn`, and `loss_name`.
#' @param folds Number of cross-validation folds (default: 3).
#' @param .seed Optional seed for reproducibility (default: 42).
#' @param refit Logical. If TRUE, refits the best model on the full dataset.
#' @param return One of "all", "summary", or "best_model":
#'   \describe{
#'     \item{"all"}{Returns the full cross-validation result across all hyperparameter combinations.}
#'     \item{"summary"}{Returns averaged results per configuration.}
#'     \item{"best_model"}{Returns the refitted model or best hyperparameters, depending on `refit`.}
#'   }
#'
#' @return A tibble or model object depending on the `return` value.
#' @export
#'
#' @examples
#' # Define hyperparameter grid
#' grid <- list(
#'   hidden     = list(c(16), c(32, 16)),
#'   lr         = c(1e-4),
#'   activation = c("relu"),
#'   epochs     = c(100),
#'   .loss_fn   = list(cox_loss, aft_loss),
#'   loss_name  = c("cox_loss", "aft_loss")
#' )
#'
#' # Load data
#' library(survival)
#' data(veteran, package = "survival")
#' eval_times <- c(90, 300)
#'
#' # (1) Return all fold-level results
#' res_all <- tune_survdnn(
#'   formula = Surv(time, status) ~ age + karno + celltype,
#'   data = veteran,
#'   times = eval_times,
#'   metrics = "cindex",
#'   param_grid = grid,
#'   folds = 3,
#'   .seed = 42,
#'   refit = FALSE,
#'   return = "all"
#' )
#'
#' # (2) Return summary per configuration (mean CV performance)
#' res_summary <- tune_survdnn(
#'   formula = Surv(time, status) ~ age + karno + celltype,
#'   data = veteran,
#'   times = eval_times,
#'   metrics = "cindex",
#'   param_grid = grid,
#'   folds = 3,
#'   .seed = 42,
#'   refit = FALSE,
#'   return = "summary"
#' )
#'
#' # (3) Return best configuration only (no refit)
#' res_best <- tune_survdnn(
#'   formula = Surv(time, status) ~ age + karno + celltype,
#'   data = veteran,
#'   times = eval_times,
#'   metrics = "cindex",
#'   param_grid = grid,
#'   folds = 3,
#'   .seed = 42,
#'   refit = FALSE,
#'   return = "best_model"
#' )
#'
#' # (4) Refit best model on full data
#' final_model <- tune_survdnn(
#'   formula = Surv(time, status) ~ age + karno + celltype,
#'   data = veteran,
#'   times = eval_times,
#'   metrics = "cindex",
#'   param_grid = grid,
#'   folds = 3,
#'   .seed = 42,
#'   refit = TRUE,
#'   return = "best_model"
#' )

tune_survdnn <- function(formula, data, times, metrics = "cindex",
                         param_grid, folds = 3, .seed = 42,
                         refit = FALSE,
                         return = c("all", "summary", "best_model")) {
  return <- match.arg(return)
  if (!is.null(.seed)) set.seed(.seed)
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
      hidden = hidden,
      lr = lr,
      activation = activation,
      epochs = epochs,
      .loss_fn = .loss_fn
    )

    dplyr::bind_cols(config_tbl[rep(1, nrow(cv_tbl)), ], cv_tbl)
  })

  summary_tbl <- summarize_tune_survdnn(all_results, by_time = FALSE)

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


#' Summarize Cross-Validation Results from tune_survdnn()
#'
#' Aggregates the results of `tune_survdnn()` to compute mean and standard deviation
#' of performance metrics per hyperparameter configuration.
#'
#' @param tuning_results A tibble returned by `tune_survdnn(return = "all")`.
#' @param by_time Logical. If `TRUE` (default), aggregates per time point as well.
#'
#' @return A tibble with mean and standard deviation of performance for each configuration.
#' @export
#'
#' @examples
#' # Example with return = "all"
#' library(survival)
#' data(veteran, package = "survival")
#' grid <- list(
#'   hidden     = list(c(16), c(32, 16)),
#'   lr         = c(1e-4),
#'   activation = c("relu"),
#'   epochs     = c(100),
#'   .loss_fn   = list(cox_loss, aft_loss),
#'   loss_name  = c("cox_loss", "aft_loss")
#' )
#' results_all <- tune_survdnn(
#'   formula = Surv(time, status) ~ age + karno + celltype,
#'   data = veteran,
#'   times = c(90, 300),
#'   metrics = "cindex",
#'   param_grid = grid,
#'   folds = 3,
#'   .seed = 42,
#'   return = "all"
#' )
#' summarize_tune_survdnn(results_all)
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
