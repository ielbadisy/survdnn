#' Build a Deep Neural Network for Survival Analysis
#'
#' Constructs a multilayer perceptron (MLP) with batch normalization,
#' activation functions, and dropout. Used internally by [survdnn()] to
#' define the model architecture.
#'
#' @param input_dim Integer. Number of input features.
#' @param hidden Integer vector. Sizes of the hidden layers (e.g., c(32, 16)).
#' @param activation Character. Name of the activation function to use in each layer.
#'   Supported options: `"relu"`, `"leaky_relu"`, `"tanh"`, `"sigmoid"`, `"gelu"`, `"elu"`, `"softplus"`.
#' @param output_dim Integer. Output layer dimension (default = 1).
#'
#' @return A `nn_sequential` object representing the network.
#' @keywords internal
#' @export
#'
#' @examples
#' \donttest{
#' net <- build_dnn(10, hidden = c(64, 32), activation = "relu")
#' }
build_dnn <- function(input_dim, hidden, activation = "relu", output_dim = 1L) {
  layers <- list()
  in_features <- input_dim

  act_fn <- switch(activation,
                   relu       = torch::nn_relu,
                   leaky_relu = torch::nn_leaky_relu,
                   tanh       = torch::nn_tanh,
                   sigmoid    = torch::nn_sigmoid,
                   gelu       = torch::nn_gelu,
                   elu        = torch::nn_elu,
                   softplus   = torch::nn_softplus,
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

  layers <- append(layers, list(torch::nn_linear(in_features, output_dim)))
  torch::nn_sequential(!!!layers)
}


#' Fit a Deep Neural Network for Survival Analysis
#'
#' Trains a deep neural network (DNN) to model right-censored survival data.
#'
#' @param formula A survival formula of the form `Surv(time, status) ~ predictors`.
#' @param data A data frame containing the variables in the model.
#' @param hidden Integer vector. Sizes of the hidden layers (default: c(32, 16)).
#' @param activation Character string specifying the activation function to use in each layer.
#'   Supported options: `"relu"`, `"leaky_relu"`, `"tanh"`, `"sigmoid"`, `"gelu"`, `"elu"`, `"softplus"`.
#' @param lr Learning rate for the Adam optimizer (default: `1e-4`).
#' @param epochs Number of training epochs (default: 300).
#' @param loss Character name of the loss function to use. One of `"cox"`, `"cox_l2"`, `"aft"`, or `"coxtime"`.
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
#'   \item{loss_history}{Vector of loss values per epoch.}
#'   \item{final_loss}{Final training loss.}
#'   \item{loss}{Loss function name used ("cox", "aft", etc.).}
#'   \item{activation}{Activation function used.}
#'   \item{hidden}{Hidden layer sizes.}
#'   \item{lr}{Learning rate.}
#'   \item{epochs}{Number of training epochs.}
#' }
#'
#' @export
survdnn <- function(formula, data,
                    hidden = c(32L, 16L),
                    activation = "relu",
                    lr = 1e-4,
                    epochs = 300L,
                    loss = c("cox", "cox_l2", "aft", "coxtime"),
                    optimizer = c("adam", "adamw", "sgd", "rmsprop", "adagrad"),
                    optim_args = list(),
                    verbose = TRUE,
                    dropout = 0.3,
                    batch_norm = TRUE,
                    callbacks = NULL,
                    .seed = NULL,
                    .device = c("auto", "cpu", "cuda")) {

  survdnn_set_seed(.seed)

  device <- survdnn_get_device(.device)

  loss      <- match.arg(loss)
  optimizer <- match.arg(optimizer)

  if (!is.list(optim_args)) {
    stop("`optim_args` must be a list (possibly empty).", call. = FALSE)
  }

  if (!is.null(callbacks)) {
    if (is.function(callbacks)) {
      callbacks <- list(callbacks)
    } else if (!is.list(callbacks) || !all(vapply(callbacks, is.function, logical(1)))) {
      stop("`callbacks` must be NULL, a function, or a list of functions.", call. = FALSE)
    }
  }

  stopifnot(inherits(formula, "formula"))
  stopifnot(is.data.frame(data))

  loss_fn <- switch(
    loss,
    cox     = cox_loss,
    cox_l2  = function(pred, true) cox_l2_loss(pred, true, lambda = 1e-3),
    aft     = aft_loss,
    coxtime = coxtime_loss
  )

  environment(formula) <- list2env(
    list(Surv = survival::Surv),
    parent = environment(formula)
  )

  mf        <- model.frame(formula, data)
  y         <- model.response(mf)
  x         <- model.matrix(attr(mf, "terms"), data = mf)[, -1, drop = FALSE]
  time      <- y[, "time"]
  status    <- y[, "status"]
  x_scaled  <- scale(x)

  x_tensor <- if (loss == "coxtime") {
    torch::torch_tensor(cbind(time, x_scaled), dtype = torch::torch_float())
  } else {
    torch::torch_tensor(
      x_scaled,
      dtype  = torch::torch_float(),
      device = device
    )
  }

  y_tensor <- torch::torch_tensor(
    cbind(time, status),
    dtype  = torch::torch_float(),
    device = device
  )

  ## build network with dropout + batch_norm controls
  net <- build_dnn(
    input_dim  = ncol(x_tensor),
    hidden     = hidden,
    activation = activation,
    output_dim = 1L,
    dropout    = dropout,
    batch_norm = batch_norm
  )
  net$to(device = device)

  ## build optimizer with dispatcher
  opt_args <- c(
    list(params = net$parameters, lr = lr),
    optim_args
  )

  ## default weight decay for adam/adamw if not provided
  if (is.null(optim_args$weight_decay) && optimizer %in% c("adam", "adamw")) {
    opt_args$weight_decay <- 1e-4
  }

  y_tensor <- torch::torch_tensor(cbind(time, status), dtype = torch::torch_float())
  net <- build_dnn(ncol(x_tensor), hidden, activation)
  optimizer <- torch::optim_adam(net$parameters, lr = lr, weight_decay = 1e-4)

  loss_history <- numeric(epochs)
  for (epoch in 1:epochs) {
    net$train()
    optimizer_obj$zero_grad()

    pred     <- net(x_tensor)
    loss_val <- loss_fn(pred, y_tensor)
    loss_val$backward()
    optimizer_obj$step()

    current_loss        <- loss_val$item()
    loss_history[epoch] <- current_loss
    last_epoch_run      <- epoch

    if (verbose && epoch %% 50 == 0) {
      cat(sprintf("Epoch %d - Loss: %.6f\n", epoch, current_loss))
      cat("\n")
    }

    ## callbacks 
    if (!is.null(callbacks)) {
      for (cb in callbacks) {
        stop_now <- isTRUE(cb(epoch, current_loss))
        if (stop_now) {
          early_stopped <- TRUE
          break
        }
      }
      if (early_stopped) break
    }
  }

  ## truncate loss history if early stopping
  if (early_stopped && last_epoch_run < epochs) {
    loss_history <- loss_history[seq_len(last_epoch_run)]
  }

  structure(
    list(
      model        = net,
      formula      = formula,
      data         = data,
      xnames       = colnames(x),
      x_center     = attr(x_scaled, "scaled:center"),
      x_scale      = attr(x_scaled, "scaled:scale"),
      loss_history = loss_history,
      final_loss   = tail(loss_history, 1),
      loss         = loss,
      activation   = activation,
      hidden       = hidden,
      lr           = lr,
      epochs       = epochs,
      optimizer    = optimizer,
      optim_args   = optim_args,
      device       = device,
      dropout      = dropout,
      batch_norm   = batch_norm
    ),
    class = "survdnn"
  )
}

