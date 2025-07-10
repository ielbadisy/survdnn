print.survdnn <- function(x, ...) {
  cat("== survdnn model ==\n")
  cat("Formula: "); print(x$formula)
  cat(sprintf("Hidden layers: %s\n", paste(x$hidden, collapse = " → ")))
  cat(sprintf("Final loss: %.4f (after %d epochs)\n", x$loss, x$epochs))
  invisible(x)
}


mod <- survdnn(Surv(time, status) ~ age + karno + celltype, data = veteran)
print(mod)
