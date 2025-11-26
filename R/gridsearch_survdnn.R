#' Grid Search for survdnn Hyperparameters
#'
#' Performs grid search over user-specified hyperparameters and evaluates performance on a validation set.
#'
#' @param formula A survival formula (e.g., `Surv(time, status) ~ .`)
#' @param train Training dataset
#' @param valid Validation dataset
#' @param times Evaluation time points (numeric vector)
#' @param metrics Evaluation metrics (character vector): any of "cindex", "ibs", "brier"
#' @param param_grid A named list of hyperparameters (`hidden`, `lr`, `activation`, `epochs`, `loss`)
#' @param .seed Optional random seed for reproducibility
#'
#' @return A tibble with configurations and their validation metrics
#' @export
#'
#' @examples
#' \donttest{
#' library(survdnn)
#' library(survival)
#' set.seed(123)
#' # Simulate small dataset
#' n <- 300
#' x1 <- rnorm(n); x2 <- rbinom(n, 1, 0.5)
#' time <- rexp(n, rate = 0.1)
#' status <- rbinom(n, 1, 0.7)
#' df <- data.frame(time, status, x1, x2)
#' # Split into training and validation
#' idx <- sample(seq_len(n), 0.7 * n)
#' train <- df[idx, ]
#' valid <- df[-idx, ]
#' # Define formula and param grid
#' formula <- Surv(time, status) ~ x1 + x2
#' param_grid <- list(
#'   hidden     = list(c(16, 8), c(32, 16)),
#'   lr         = c(1e-3),
#'   activation = c("relu"),
#'   epochs     = c(100),
#'   loss       = c("cox", "coxtime")
#' )
#' # Run grid search
#' results <- gridsearch_survdnn(
#'   formula = formula,
#'   train   = train,
#'   valid   = valid,
#'   times   = c(10, 20, 30),
#'   metrics = c("cindex", "ibs"),
#'   param_grid = param_grid
#' )
#' # View summary
#' dplyr::group_by(results, hidden, lr, activation, epochs, loss, metric) |>
#'   dplyr::summarise(mean = mean(value, na.rm = TRUE), .groups = "drop")
#' }
gridsearch_survdnn <- function(formula, train, valid, times,
                               metrics = c("cindex", "ibs"),
                               param_grid, .seed = 42) {
  if (!is.null(.seed)) set.seed(.seed)
  param_df <- tidyr::crossing(!!!param_grid)

  results <- purrr::pmap_dfr(param_df, function(hidden, lr, activation, epochs, loss) {
    message(glue::glue("[survdnn] Training: loss={loss}, activation={activation}, hidden={toString(hidden)}"))

    mod <- survdnn(
      formula    = formula,
      data       = train,
      hidden     = hidden,
      lr         = lr,
      activation = activation,
      epochs     = epochs,
      loss       = loss,
      verbose    = FALSE
    )

    eval_tbl <- evaluate_survdnn(
      model    = mod,
      newdata  = valid,
      metrics  = metrics,
      times    = times
    )

    config <- tibble::tibble(
      hidden     = list(hidden),
      lr         = lr,
      activation = activation,
      epochs     = epochs,
      loss       = loss
    )

    dplyr::bind_cols(config[rep(1, nrow(eval_tbl)), ], eval_tbl)
  })

  return(results)
}
