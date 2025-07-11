print.survdnn <- function(x, ...) {
  cli::cli_h1("survdnn model")
  cli::cli_text("Formula: {.val {deparse(x$formula)}}")
  cli::cli_text("Hidden layers: {.val {paste(x$hidden, collapse = ' → ')}}")
  if (!is.null(x$lr))         cli::cli_text("Learning rate: {.val {x$lr}}")
  if (!is.null(x$activation)) cli::cli_text("Activation: {.val {x$activation}}")
  if (!is.null(x$loss_name))  cli::cli_text("Loss function: {.val {x$loss_name}}")
  cli::cli_text("Final loss: {.val {round(x$loss, 4)}} after {.val {x$epochs}} epochs")

  invisible(x)
}

library(survival)
data(veteran)
mod <- survdnn(Surv(time, status) ~ age + karno + celltype, data = veteran)
print(mod)
