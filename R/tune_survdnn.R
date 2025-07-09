
tune_survdnn <- function(formula, data, times, metrics = "cindex",
                         param_grid, folds = 3, seed = 42) {
  if (!requireNamespace("dplyr", quietly = TRUE) ||
      !requireNamespace("tibble", quietly = TRUE) ||
      !requireNamespace("purrr", quietly = TRUE)) {
    stop("Packages 'dplyr', 'tibble', and 'purrr' are required.")
  }

  param_df <- tidyr::crossing(!!!param_grid)

  purrr::pmap_dfr(param_df, function(hidden, lr, activation, epochs) {
    config_tbl <- tibble::tibble(
      hidden = list(hidden),
      lr = lr,
      activation = activation,
      epochs = epochs
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
  }) |>
    dplyr::arrange(metric, dplyr::desc(value))
}

library(survival)
data(veteran)

grid <- list(
  hidden = list(c(16), c(32, 16)),
  lr = c(1e-4, 5e-4),
  activation = c("relu", "tanh"),
  epochs = c(300)
)

tune_res <- tune_survdnn(
  formula = Surv(time, status) ~ age + karno + celltype,
  data = veteran,
  times = c(90),
  metrics = c("cindex", "ibs"),
  param_grid = grid,
  folds = 3,
  seed = 42
)
print(tune_res)

