.onLoad <- function(libname, pkgname) {
  
  # package level option
  op <- options()
  op.survdnn <- list(survdnn.default_epochs = 100)
  toset <- !(names(op.survdnn) %in% names(op))
  if (any(toset)) options(op.survdnn[toset])
  
}



