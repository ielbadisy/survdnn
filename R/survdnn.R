library(torch)

build_dnn <- function(input_dim, hidden, activation = "relu") {
  layers <- list()
  in_features <- input_dim

  act_fn <- switch(activation,
    relu        = torch::nn_relu,
    leaky_relu  = torch::nn_leaky_relu,
    tanh        = torch::nn_tanh,
    sigmoid     = torch::nn_sigmoid,
    gelu        = torch::nn_gelu,
    elu         = torch::nn_elu,
    softplus    = torch::nn_softplus,
                   stop("Unsupported activation function: ", activation)
  )

  for (h in hidden) {
    layers <- append(layers, list(
      torch::nn_linear(in_features, h),
      torch::nn_batch_norm1d(h),
      act_fn(),
      torch::nn_dropout(p = 0.3)
    ))
    in_features <- h
  }

  layers <- append(layers, list(torch::nn_linear(in_features, 1)))
  torch::nn_sequential(!!!layers)
}

cox_loss <- function(pred, true) {
  time <- true[, 1]
  status <- true[, 2]
  idx <- torch_argsort(time, descending = TRUE)
  time <- time[idx]
  status <- status[idx]
  pred <- -pred[idx, 1]  

  log_cumsum_exp <- torch_logcumsumexp(pred, dim = 1)
  event_mask <- (status == 1)
  loss <- -torch_mean(pred[event_mask] - log_cumsum_exp[event_mask])
  return(loss)
}

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

# store loss at each epoch
loss_history <- numeric(epochs)

for (epoch in 1:epochs) {
net$train()
optimizer$zero_grad()
pred <- net(x_tensor)
loss <- .loss_fn(pred, y_tensor)
loss$backward()
optimizer$step()

loss_history[epoch] <- loss$item()

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
loss = tail(loss_history, 1),
loss_history = loss_history,  # track the loss history
activation = activation,
hidden = hidden,
lr = lr,
epochs = epochs,
.loss_fn = .loss_fn,
loss_name = deparse(substitute(.loss_fn))  # fixed naming
), class = "survdnn")
}
