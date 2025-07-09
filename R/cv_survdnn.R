cv_survdnn <- function(formula, data, times,
                       metrics = c("cindex", "ibs"),
                       folds = 5, seed = NULL, ...) {
  if (!requireNamespace("rsample", quietly = TRUE)) {
    stop("Package 'rsample' is required for cross-validation.")
  }

  if (!inherits(formula, "formula")) stop("`formula` must be a survival formula.")
  if (!is.data.frame(data)) stop("`data` must be a data frame.")
  if (missing(times)) stop("You must provide a `times` vector.")

  if (!is.null(seed)) set.seed(seed)
  vfolds <- rsample::vfold_cv(data, v = folds, strata = all.vars(formula)[1])

  results <- purrr::imap_dfr(vfolds$splits, function(split, i) {
    train_data <- rsample::analysis(split)
    test_data  <- rsample::assessment(split)

    model <- survdnn(formula, data = train_data, ...)

    eval_tbl <- evaluate_survdnn(model, metrics = metrics, times = times, newdata = test_data)
    eval_tbl$fold <- i
    eval_tbl
  })

  dplyr::select(results, fold, metric, time = dplyr::any_of("time"), value)
}



library(survival)
data(veteran)

cv_res <- cv_survdnn(
  formula = Surv(time, status) ~ age + karno + celltype,
  data = veteran,
  times = c(30, 90, 180),
  metrics = c("cindex", "ibs"),
  folds = 3,
  seed = 42,
  hidden = c(16, 8), epochs = 200
)

cv_res
