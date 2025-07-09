
survdnn <- function(formula, data,
                    hidden = c(32L, 16L),
                    activation = "relu",
                    lr = 1e-4,
                    epochs = 300L,
                    .loss_fn = cox_loss,
                    verbose = TRUE) {

  stopifnot(inherits(formula, "formula"))
  stopifnot(is.data.frame(data))
  if (!is.function(.loss_fn)) stop("`.loss_fn` must be a function.")

  mf <- model.frame(formula, data)
  y <- model.response(mf)
  x <- model.matrix(attr(mf, "terms"), data)[, -1, drop = FALSE]

  time <- y[, "time"]
  status <- y[, "status"]
  x_scaled <- scale(x)

  x_tensor <- torch::torch_tensor(as.matrix(x_scaled), dtype = torch::torch_float())
  y_tensor <- torch::torch_tensor(as.matrix(cbind(time, status)), dtype = torch::torch_float())

  net <- build_dnn(ncol(x_tensor), hidden, activation)
  optimizer <- torch::optim_adam(net$parameters, lr = lr, weight_decay = 1e-4)

  for (epoch in 1:epochs) {
    net$train()
    optimizer$zero_grad()
    pred <- net(x_tensor)
    loss <- .loss_fn(pred, y_tensor)
    loss$backward()
    optimizer$step()
    if (verbose && epoch %% 50 == 0) {
      cat(sprintf("Epoch %d - Loss: %.6f\n", epoch, loss$item()))
    }
  }

  structure(list(
    model = net,
    formula = formula,
    data = data,
    xnames = colnames(x),
    x_center = attr(x_scaled, "scaled:center"),
    x_scale = attr(x_scaled, "scaled:scale"),
    loss = loss$item(),
    activation = activation,
    hidden = hidden,
    lr = lr,
    epochs = epochs,
    .loss_fn = .loss_fn
  ), class = "survdnn", engine = "torchsurv")
}
