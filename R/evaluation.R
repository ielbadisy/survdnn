#' Evaluate a survdnn Model Using Survival Metrics
#'
#' Computes evaluation metrics for a fitted `survdnn` model at one or more time points.
#' Supported metrics include the concordance index (`"cindex"`), Brier score (`"brier"`),
#' and integrated Brier score (`"ibs"`).
#'
#' @param model A fitted `survdnn` model object.
#' @param metrics A character vector of metric names: `"cindex"`, `"brier"`, `"ibs"`.
#' @param times A numeric vector of evaluation time points.
#' @param newdata Optional. A data frame on which to evaluate the model. Defaults to training data.
#' @param na_action Character. How to handle missing values in evaluation data:
#'   `"omit"` drops incomplete rows, `"fail"` errors if any NA is present.
#' @param verbose Logical. If TRUE and `na_action="omit"`, prints a message when rows are removed.
#'
#' @return A tibble with evaluation results, containing at least `metric`, `value`, and possibly `time`.
#' @export
evaluate_survdnn <- function(model,
                             metrics = c("cindex", "brier", "ibs"),
                             times,
                             newdata = NULL,
                             na_action = c("omit", "fail"),
                             verbose = FALSE) {
  
  stopifnot(inherits(model, "survdnn"))
  if (missing(times)) stop("You must provide `times` for evaluation.", call. = FALSE)

  na_action <- match.arg(na_action)

  allowed_metrics <- c("cindex", "brier", "ibs")
  unknown <- setdiff(metrics, allowed_metrics)
  if (length(unknown) > 0) {
    stop("Unknown metric(s): ", paste(unknown, collapse = ", "), call. = FALSE)
  }
 
  data <- if (is.null(newdata)) model$data else newdata
  n_before <- nrow(data)
  
  # build model frame first with explicit NA policy
  mf <- model.frame(
    model$formula,
    data = data,
    na.action = if (na_action == "omit") stats::na.omit else stats::na.fail
  )
  
  n_after <- nrow(mf)
  n_removed <- n_before - n_after
  if (n_removed > 0 && isTRUE(verbose) && na_action == "omit") {
    message(sprintf("[survdnn::eval] removed %d observations with missing values in evaluation data.", n_removed))
  }

  y <- model.response(mf)
  if (!inherits(y, "Surv")) stop("The response must be a 'Surv' object.", call. = FALSE)

  # predict on the filtered mf to keep row alignment
  sp_matrix <- predict(model, newdata = mf, times = times, type = "survival")

  purrr::map_dfr(metrics, function(metric) { ## (to replace map_dfr() with fmap() from functionals package)
    if (metric == "brier" && length(times) > 1) {
      tibble::tibble(
        metric = "brier",
        time = times,
        value = vapply(seq_along(times), function(i) {
          brier(y, pre_sp = sp_matrix[, i], t_star = times[i])
        }, numeric(1))
      )
    } else {
      val <- switch(
        metric,
        "cindex" = cindex_survmat(y, predicted = sp_matrix, t_star = max(times)),
        "brier"  = brier(y, pre_sp = sp_matrix[, 1], t_star = times[1]),
        "ibs"    = ibs_survmat(y, sp_matrix, times)
      )
      tibble::tibble(metric = metric, value = val)
    }
  })
}


#' K-Fold Cross-Validation for survdnn Models
#'
#' Performs cross-validation for a `survdnn` model using the specified evaluation metrics.
#'
#' @param formula A survival formula, e.g., `Surv(time, status) ~ x1 + x2`.
#' @param data A data frame.
#' @param times A numeric vector of evaluation time points.
#' @param metrics A character vector: any of `"cindex"`, `"brier"`, `"ibs"`.
#' @param folds Integer. Number of folds to use.
#' @param .seed Optional. Set random seed for reproducibility.
#' @param .device Character string indicating the computation device used when fitting the models
#'   in each fold. One of `"auto"`, `"cpu"`, or `"cuda"`. `"auto"` uses CUDA if available,
#'   otherwise falls back to CPU.
#' @param .threads Optional positive integer. If provided, sets Torch CPU thread
#'   count before each fold fit via `torch::torch_set_num_threads()`.
#' @param na_action Character. How to handle missing values within each fold:
#'   `"omit"` drops incomplete rows; `"fail"` errors if any NA is present.
#' @param verbose Logical; whether to print cross-validation progress and propagate
#'   verbose messages to fitting/evaluation in each fold (default: TRUE).
#' @param ... Additional arguments passed to [survdnn()].
#'
#' @return A tibble containing metric values per fold and (optionally) per time point.
#' @export
#'
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
#'   veteran <- survival::veteran
#'   cv_survdnn(
#'     survival::Surv(time, status) ~ age + karno + celltype,
#'     data = veteran,
#'     times = c(30, 90, 180),
#'     metrics = "ibs",
#'     folds = 3,
#'     .seed = 42,
#'     hidden = c(16, 8),
#'     epochs = 5
#'   )
#' }
#' }
cv_survdnn <- function(formula, data, times,
                       metrics = c("cindex", "ibs"),
                       folds = 5,
                       .seed = NULL,
                       .device = c("auto", "cpu", "cuda"),
                       .threads = NULL,
                       na_action = c("omit", "fail"),
                       verbose = TRUE,
                       ...) {

  .device   <- match.arg(.device)
  na_action <- match.arg(na_action)

  if (!requireNamespace("rsample", quietly = TRUE)) {
    stop("Package 'rsample' is required for cross-validation.", call. = FALSE)
  }

  if (!inherits(formula, "formula")) stop("`formula` must be a survival formula", call. = FALSE)
  if (!is.data.frame(data)) stop("`data` must be a data frame", call. = FALSE)
  if (missing(times)) stop("You must provide a `times` vector.", call. = FALSE)

  if (!is.null(.seed)) survdnn_set_seed(.seed)

  if (isTRUE(verbose)) {
    message(
      sprintf(
        "[survdnn::cv] start: folds=%d n=%d metrics=%s times=%s",
        folds,
        nrow(data),
        paste(metrics, collapse = ","),
        paste(times, collapse = ",")
      )
    )
    if (!is.null(.threads)) {
      message(sprintf("[survdnn::cv] cpu_threads=%d", as.integer(.threads)))
    }
  }

  status_var <- all.vars(formula[[2]])[2]          # more safe for extracting the status
  vfolds <- rsample::vfold_cv(data, v = folds, strata = dplyr::all_of(status_var))

  results <- purrr::imap_dfr(vfolds$splits, function(split, i) {

    ## re-seed inside every fold to ensure full reproducibility
    survdnn_set_seed(.seed)

    train_data <- rsample::analysis(split)
    test_data  <- rsample::assessment(split)

    if (isTRUE(verbose)) {
      message(
        sprintf(
          "[survdnn::cv] fold %d/%d: train_n=%d test_n=%d",
          i, folds, nrow(train_data), nrow(test_data)
        )
      )
    }

    model <- survdnn(
      formula,
      data      = train_data,
      verbose   = verbose,
      .seed     = .seed,
      .device   = .device,
      .threads  = .threads,
      na_action = na_action,
      ...
    )

    eval_tbl <- evaluate_survdnn(
      model,
      metrics   = metrics,
      times     = times,
      newdata   = test_data,
      na_action = na_action,
      verbose   = verbose
    )

    eval_tbl$fold <- i
    if (isTRUE(verbose)) {
      message(sprintf("[survdnn::cv] fold %d/%d done.", i, folds))
    }
    eval_tbl
  })

  if (isTRUE(verbose)) {
    message(sprintf("[survdnn::cv] done: completed %d folds.", folds))
  }

  dplyr::select(results, fold, metric, time = dplyr::any_of("time"), value)
}


#' Summarize Cross-Validation Results from survdnn
#'
#' Computes mean, standard deviation, and confidence intervals for metrics from cross-validation.
#'
#' @param cv_results A tibble returned by [cv_survdnn()].
#' @param by_time Logical. Whether to stratify results by `time` (if present).
#' @param conf_level Confidence level for the intervals (default: 0.95).
#'
#' @return A tibble summarizing mean, sd, and confidence bounds per metric (and per time if applicable).
#' @export
#'
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
#'   veteran <- survival::veteran
#'   res <- cv_survdnn(
#'     survival::Surv(time, status) ~ age + karno + celltype,
#'     data = veteran,
#'     times = c(30, 90, 180, 270),
#'     metrics = c("cindex", "ibs"),
#'     folds = 3,
#'     .seed = 42,
#'     hidden = c(16, 8),
#'     epochs = 5
#'   )
#'   summarize_cv_survdnn(res)
#' }
#' }
summarize_cv_survdnn <- function(cv_results, by_time = TRUE, conf_level = 0.95) {
  stopifnot(all(c("fold", "metric", "value") %in% names(cv_results)))

  z <- qnorm((1 + conf_level) / 2)
  group_vars <- if ("time" %in% names(cv_results) && by_time) {
    c("metric", "time")
  } else {
    "metric"
  }

  cv_results |>
    dplyr::group_by(dplyr::across(all_of(group_vars))) |>
    dplyr::summarize(
      mean = mean(value, na.rm = TRUE),
      sd = sd(value, na.rm = TRUE),
      n = dplyr::n(),
      se = sd / sqrt(n),
      lower = mean - z * se,
      upper = mean + z * se,
      .groups = "drop"
    ) |>
    dplyr::select(-n, -se)
}
