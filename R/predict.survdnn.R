
predict.survdnn <- function(object, newdata, times = NULL,
                            type = c("survival", "lp", "risk"), ...) {
  type <- match.arg(type)
  stopifnot(inherits(object, "survdnn"))
  stopifnot(is.data.frame(newdata))

  # rebuild predictor matrix
  x <- model.matrix(delete.response(terms(object$formula)), newdata)[, object$xnames, drop = FALSE]
  x_scaled <- scale(x, center = object$x_center, scale = object$x_scale)
  x_tensor <- torch::torch_tensor(as.matrix(x_scaled), dtype = torch::torch_float())

  # predict linear predictor (negative risk score)
  object$model$eval()
  with_no_grad({
    lp <- -as.numeric(object$model(x_tensor)[, 1])
  })

  if (type == "lp") return(lp)

  if (is.null(times)) stop("`times` must be specified for type = 'survival' or 'risk'.")
  times <- sort(times)

  # estimate baseline hazard using training set
  train_x <- model.matrix(delete.response(terms(object$formula)), object$data)[, object$xnames, drop = FALSE]
  train_x_scaled <- scale(train_x, center = object$x_center, scale = object$x_scale)
  train_lp <- -as.numeric(object$model(torch::torch_tensor(train_x_scaled, dtype = torch::torch_float()))[, 1])
  train_df <- data.frame(
    time = model.response(model.frame(object$formula, object$data))[, "time"],
    status = model.response(model.frame(object$formula, object$data))[, "status"],
    lp = train_lp
  )
  bh <- survival::basehaz(survival::coxph(Surv(time, status) ~ lp, data = train_df), centered = FALSE)
  H0 <- approx(bh$time, bh$hazard, xout = times, rule = 2)$y

  # compute survival matrix
  lp_clipped <- pmin(pmax(lp, -3), 3)
  surv_mat <- outer(lp_clipped, H0, function(lp_i, h0_j) exp(-h0_j * exp(lp_i)))

  if (type == "risk") {
    return(1 - surv_mat[, length(times)])
  }

  surv_df <- as.data.frame(surv_mat)
  colnames(surv_df) <- paste0("t=", times)
  return(surv_df)
}



# TEST

library(survival)
data(veteran)

mod <- survdnn(Surv(time, status) ~ age + karno + celltype, data = veteran)
pred_lp <- predict(mod, veteran, type = "lp")
pred_surv <- predict(mod, veteran, type = "survival", times = c(30, 90, 180))
pred_risk <- predict(mod, veteran, type = "risk", times = 180)
