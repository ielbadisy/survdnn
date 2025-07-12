#' Build a Deep Neural Network for Survival Analysis
#'
#' Constructs a multilayer perceptron (MLP) with batch normalization,
#' activation functions, and dropout. Used internally by `survdnn()` to
#' define the model architecture.
#'
#' @param input_dim Integer. Number of input features.
#' @param hidden Integer vector. Sizes of the hidden layers (e.g., c(32, 16)).
#' @param activation Character. Name of the activation function to use in each layer.
#'   Supported options: `"relu"`, `"leaky_relu"`, `"tanh"`, `"sigmoid"`, `"gelu"`, `"elu"`, `"softplus"`.
#'
#' @return A `nn_sequential` object representing the network.
#' @keywords internal
#'
#' @examples
#' # Internal use only
#' net <- build_dnn(10, hidden = c(64, 32), activation = "relu")

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


#' Fit a Deep Neural Network for Survival Analysis
#'
#' Trains a flexible deep neural network (DNN) to model right-censored survival data
#' using the specified loss function. Supports Cox loss and any custom `torch`-compatible loss.
#'
#' @param formula A survival formula of the form `Surv(time, status) ~ predictors`.
#' @param data A data.frame containing the variables in the model.
#' @param hidden Integer vector. Sizes of the hidden layers (default: c(32, 16)).
#' @param activation Character string specifying the activation function to use in each layer.
#'   Supported: `"relu"`, `"leaky_relu"`, `"tanh"`, `"sigmoid"`, `"gelu"`, `"elu"`, `"softplus"`.
#' @param lr Learning rate for the Adam optimizer (default: `1e-4`).
#' @param epochs Number of training epochs (default: 300).
#' @param .loss_fn A custom loss function compatible with torch (default: `cox_loss`).
#' @param verbose Logical; whether to print loss progress every 50 epochs (default: TRUE).
#'
#' @return An object of class `"survdnn"` containing:
#' \describe{
#'   \item{model}{Trained `nn_module` object.}
#'   \item{formula}{Original survival formula.}
#'   \item{data}{Training data used for fitting.}
#'   \item{xnames}{Predictor variable names.}
#'   \item{x_center}{Column means of predictors.}
#'   \item{x_scale}{Column standard deviations of predictors.}
#'   \item{loss}{Final epoch loss.}
#'   \item{loss_history}{Vector of loss values per epoch.}
#'   \item{activation}{Used activation function.}
#'   \item{hidden}{Hidden layer sizes.}
#'   \item{lr}{Learning rate used.}
#'   \item{epochs}{Number of training epochs.}
#'   \item{.loss_fn}{The loss function used for training.}
#'   \item{loss_name}{Character representation of the loss function name.}
#' }
#'
#' @export
#'
#' @examples
#' set.seed(123)
#' n <- 100
#' x1 <- rnorm(n)
#' x2 <- rbinom(n, 1, 0.5)
#' time <- rexp(n, rate = 0.1)
#' status <- rbinom(n, 1, 0.7)
#' df <- data.frame(time = time, status = status, x1 = x1, x2 = x2)
#'
#' mod <- survdnn(Surv(time, status) ~ ., data = df, epochs = 50, verbose = FALSE)
#' mod$loss  # final training loss

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
