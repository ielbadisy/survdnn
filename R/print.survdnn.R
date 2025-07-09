
print.survdnn <- function(x, ...) {
  stopifnot(inherits(x, "survdnn"))

  cat("== survdnn model (torch-based DNN for survival) ==\n")
  cat("• Formula: "); print(x$formula)
  cat("• Hidden layers: ", paste(x$hidden, collapse = " → "), "\n", sep = "")
  cat("• Activation: ", x$activation, "\n", sep = "")
  cat("• Epochs trained: ", x$epochs, "\n", sep = "")
  cat("• Learning rate: ", x$lr, "\n", sep = "")
  cat("• Final loss: ", formatC(x$loss, digits = 6, format = "f"), "\n", sep = "")

  invisible(x)
}


mod <- survdnn(Surv(time, status) ~ age + karno + celltype, data = veteran)
print(mod)
