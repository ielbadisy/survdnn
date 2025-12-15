utils::globalVariables(c("loss", "epoch"))

#' Tune Hyperparameters for a survdnn Model via Cross-Validation
#'
#' Performs k-fold cross-validation over a user-defined hyperparameter grid
#' and selects the best configuration according to the specified evaluation metric.
#'
#' @param formula A survival formula, e.g., `Surv(time, status) ~ x1 + x2`.
#' @param data A data frame.
#' @param times A numeric vector of evaluation time points.
#' @param metrics A character vector of evaluation metrics: "cindex", "brier", or "ibs".
#'   Only the first metric is used for model selection.
#' @param param_grid A named list defining hyperparameter combinations to evaluate.
#'   Required names: `hidden`, `lr`, `activation`, `epochs`, `loss`.
#' @param folds Number of cross-validation folds (default: 3).
#' @param .seed Optional seed for reproducibility.
#' @param .device Character string indicating the computation device used when fitting models
#'   during cross-validation and refitting. One of `"auto"`, `"cpu"`, or `"cuda"`. `"auto"`
#'   uses CUDA if available, otherwise falls back to CPU.
#' @param na_action Character. How to handle missing values:
#'   `"omit"` drops incomplete rows; `"fail"` errors if any NA is present.
#' @param refit Logical. If TRUE, refits the best model on the full dataset.
#' @param return One of "all", "summary", or "best_model":
#'   \describe{
#'     \item{"all"}{Returns the full cross-validation result across all combinations.}
#'     \item{"summary"}{Returns averaged results per configuration.}
#'     \item{"best_model"}{Returns the refitted model or best hyperparameters.}
#'   }
#'
#' @return A tibble or model object depending on the `return` value.
#' @export
tune_survdnn <- function(formula,
                         data,
                         times,
                         metrics = "cindex",
                         param_grid,
                         folds = 3,
                         .seed = 42,
                         .device = c("auto", "cpu", "cuda"),
                         na_action = c("omit", "fail"),
                         refit = FALSE,
                         return = c("all", "summary", "best_model")) {

  return    <- match.arg(return)
  .device   <- match.arg(.device)
  na_action <- match.arg(na_action)

  if (!is.null(.seed)) survdnn_set_seed(.seed)

  param_df <- tidyr::crossing(!!!param_grid)

  all_results <- purrr::pmap_dfr(
    param_df,
    function(hidden, lr, activation, epochs, loss) {

      config_tbl <- tibble::tibble(
        hidden     = list(hidden),
        lr         = lr,
        activation = activation,
        epochs     = epochs,
        loss       = loss
      )

      cv_tbl <- cv_survdnn(
        formula   = formula,
        data      = data,
        times     = times,
        metrics   = metrics,
        folds     = folds,
        hidden    = hidden,
        lr        = lr,
        activation = activation,
        epochs     = epochs,
        loss       = loss,
        .seed      = .seed,
        .device    = .device,
        na_action  = na_action
      )

      dplyr::bind_cols(config_tbl[rep(1, nrow(cv_tbl)), ], cv_tbl)
    }
  )

  summary_tbl <- summarize_tune_survdnn(all_results, by_time = FALSE)

  ## Select best hyperparameters
  primary_metric <- metrics[1]

  best_row_all <- all_results |>
    dplyr::filter(metric == primary_metric) |>
    dplyr::group_by(hidden, lr, activation, epochs, loss) |>
    dplyr::summarise(mean = mean(value, na.rm = TRUE), .groups = "drop") |>
    dplyr::slice_max(
      order_by = if (primary_metric == "cindex") mean else -mean,
      n = 1
    )

  if (nrow(best_row_all) == 0) {
    stop("No valid configuration found for primary metric: ", primary_metric, call. = FALSE)
  }

  ## Refitting the best model
  if (refit) {
    message("Refitting best model on full data...")
    best_model <- survdnn(
      formula    = formula,
      data       = data,
      hidden     = best_row_all$hidden[[1]],
      lr         = best_row_all$lr,
      activation = best_row_all$activation,
      epochs     = best_row_all$epochs,
      loss       = best_row_all$loss[[1]],
      .seed      = .seed,
      .device    = .device,
      na_action  = na_action
    )
  }

  switch(
    return,
    "all"        = all_results,
    "summary"    = summary_tbl,
    "best_model" = if (refit) best_model else dplyr::select(best_row_all, -mean)
  )
}


#' Summarize survdnn Tuning Results
#'
#' Aggregates cross-validation results from `tune_survdnn(return = "all")`
#' by configuration, metric, and optionally by time point.
#'
#' @param tuning_results The full tibble returned by `tune_survdnn(..., return = "all")`.
#' @param by_time Logical; whether to group and summarize separately by time points.
#'
#' @return A summarized tibble with mean and standard deviation of performance metrics.
#' @export
summarize_tune_survdnn <- function(tuning_results, by_time = TRUE) {
  if (!all(c("metric", "value") %in% names(tuning_results))) {
    stop("Input must be the result of `tune_survdnn(return = 'all')`.", call. = FALSE)
  }

  group_vars <- c("hidden", "lr", "activation", "epochs", "loss", "metric")
  if (by_time && "time" %in% names(tuning_results)) {
    group_vars <- c(group_vars, "time")
  }

  tuning_results |>
    dplyr::group_by(dplyr::across(all_of(group_vars))) |>
    dplyr::summarise(
      mean = mean(value, na.rm = TRUE),
      sd   = sd(value, na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::arrange(metric, dplyr::desc(mean))
}
