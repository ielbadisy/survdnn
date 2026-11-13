#' Deprecated functions
#'
#' These underscore names are kept for one release and will be removed in the next.
#' Use the glued lowercase names instead.
#'
#' @param ... Passed to the new function.
#' @return The value of the new function.
#' @name survdnn-deprecated
#' @keywords internal
NULL

#' @rdname survdnn-deprecated
#' @export
aft_loss <- function(...) {
  .Deprecated("aftloss", package = "survdnn")
  aftloss(...)
}

#' @rdname survdnn-deprecated
#' @export
build_dnn <- function(...) {
  .Deprecated("builddnn", package = "survdnn")
  builddnn(...)
}

#' @rdname survdnn-deprecated
#' @export
callback_early_stopping <- function(...) {
  .Deprecated("callbackearlystopping", package = "survdnn")
  callbackearlystopping(...)
}

#' @rdname survdnn-deprecated
#' @export
cindex_survmat <- function(...) {
  .Deprecated("cindexsurvmat", package = "survdnn")
  cindexsurvmat(...)
}

#' @rdname survdnn-deprecated
#' @export
cox_l2_loss <- function(...) {
  .Deprecated("coxl2loss", package = "survdnn")
  coxl2loss(...)
}

#' @rdname survdnn-deprecated
#' @export
cox_loss <- function(...) {
  .Deprecated("coxloss", package = "survdnn")
  coxloss(...)
}

#' @rdname survdnn-deprecated
#' @export
coxtime_loss <- function(...) {
  .Deprecated("coxtimeloss", package = "survdnn")
  coxtimeloss(...)
}

#' @rdname survdnn-deprecated
#' @export
cv_survdnn <- function(...) {
  .Deprecated("cvsurvdnn", package = "survdnn")
  cvsurvdnn(...)
}

#' @rdname survdnn-deprecated
#' @export
evaluate_survdnn <- function(...) {
  .Deprecated("evaluatesurvdnn", package = "survdnn")
  evaluatesurvdnn(...)
}

#' @rdname survdnn-deprecated
#' @export
gridsearch_survdnn <- function(...) {
  .Deprecated("gridsearchsurvdnn", package = "survdnn")
  gridsearchsurvdnn(...)
}

#' @rdname survdnn-deprecated
#' @export
ibs_survmat <- function(...) {
  .Deprecated("ibssurvmat", package = "survdnn")
  ibssurvmat(...)
}

#' @rdname survdnn-deprecated
#' @export
plot_loss <- function(...) {
  .Deprecated("plotloss", package = "survdnn")
  plotloss(...)
}

#' @rdname survdnn-deprecated
#' @export
summarize_cv_survdnn <- function(...) {
  .Deprecated("summarizecvsurvdnn", package = "survdnn")
  summarizecvsurvdnn(...)
}

#' @rdname survdnn-deprecated
#' @export
summarize_tune_survdnn <- function(...) {
  .Deprecated("summarizetunesurvdnn", package = "survdnn")
  summarizetunesurvdnn(...)
}

#' @rdname survdnn-deprecated
#' @export
tune_survdnn <- function(...) {
  .Deprecated("tunesurvdnn", package = "survdnn")
  tunesurvdnn(...)
}
