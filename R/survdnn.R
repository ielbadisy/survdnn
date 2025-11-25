
#' Build a Deep Neural Network for Survival Analysis
#'
#' Constructs a multilayer perceptron (MLP) with optional batch normalization
#' and dropout. Used internally by [survdnn()] to define the model architecture.
#'
#' @param input_dim Integer. Number of input features.
#' @param hidden Integer vector. Sizes of the hidden layers (e.g., c(32, 16)).
#' @param activation Character. Name of the activation function to use in each layer.
#'   Supported options: `"relu"`, `"leaky_relu"`, `"tanh"`, `"sigmoid"`, `"gelu"`, `"elu"`, `"softplus"`.
#' @param output_dim Integer. Output layer dimension (default = 1).
#' @param dropout Numeric between 0 and 1. Dropout rate after each hidden layer
#'   (default = 0.3). Set to 0 to disable dropout.
#' @param batch_norm Logical; whether to add `nn_batch_norm1d()` after each
#'   hidden linear layer (default = TRUE).
#'
#' @return A `nn_sequential` object representing the network.
#' @keywords internal
#' @export

build_dnn <- function(input_dim,
                      hidden,
                      activation = "relu",
                      output_dim = 1L,
                      dropout = 0.3,
                      batch_norm = TRUE) {

  layers <- list()
  in_features <- input_dim

  act_fn <- switch(
    activation,
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
    layers <- append(layers, list(torch::nn_linear(in_features, h)))
    if (isTRUE(batch_norm)) {
      layers <- append(layers, list(torch::nn_batch_norm1d(h)))
    }
    layers <- append(layers, list(act_fn()))
    if (!is.null(dropout) && dropout > 0) {
      layers <- append(layers, list(torch::nn_dropout(p = dropout)))
    }
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
#' @param activation Character string specifying the activation function.
#' @param lr Learning rate for the Adam optimizer (default: `1e-4`).
#' @param epochs Number of training epochs (default: 300).
#' @param loss Character name of the loss function to use. One of `"cox"`,
#'   `"cox_l2"`, `"aft"`, or `"coxtime"`.
#' @param verbose Logical; whether to print loss progress every 50 epochs.
#' @param dropout Numeric between 0 and 1. Dropout rate applied after each
#'   hidden layer (default = 0.3). Set to 0 to disable dropout.
#' @param batch_norm Logical; whether to apply batch normalization after each
#'   hidden layer (default = TRUE).
#' @param callbacks Optional list of callback functions to control training
#'   (e.g., early stopping).
#' @param .seed Optional integer for full reproducibility.
#' @param .device Character string: `"auto"`, `"cpu"`, or `"cuda"`.
#'
#' @return A `survdnn` model object.
#' @export

survdnn <- function(formula, data,
                    hidden = c(32L, 16L),
                    activation = "relu",
                    lr = 1e-4,
                    epochs = 300L,
                    loss = c("cox", "cox_l2", "aft", "coxtime"),
                    verbose = TRUE,
                    dropout = 0.3,
                    batch_norm = TRUE,
                    callbacks = NULL,
                    .seed = NULL,
                    .device = c("auto", "cpu", "cuda")) {


  survdnn_set_seed(.seed)

  device <- survdnn_get_device(.device)

  if (!is.null(callbacks)) {
    if (is.function(callbacks)) {
      callbacks <- list(callbacks)
    } else if (!is.list(callbacks) || !all(vapply(callbacks, is.function, logical(1)))) {
      stop("`callbacks` must be NULL, a function, or a list of functions.", call. = FALSE)
    }
  }
  

  stopifnot(inherits(formula, "formula"))
  stopifnot(is.data.frame(data))

  loss <- match.arg(loss)
  loss_fn <- switch(
    loss,
    cox     = cox_loss,
    cox_l2  = function(pred, true) cox_l2_loss(pred, true, lambda = 1e-3),
    aft     = aft_loss,
    coxtime = coxtime_loss
  )

  # Ensure Surv is resolvable in the formula environment
  environment(formula) <- list2env(
    list(Surv = survival::Surv),
    parent = environment(formula)
  )

  mf    <- model.frame(formula, data)
  y     <- model.response(mf)
  x     <- model.matrix(attr(mf, "terms"), data = mf)[, -1, drop = FALSE]
  time  <- y[, "time"]
  status <- y[, "status"]
  x_scaled <- scale(x)

  x_tensor <- if (loss == "coxtime") {
    torch::torch_tensor(
      cbind(time, x_scaled),
      dtype  = torch::torch_float(),
      device = device
    )
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

  net <- build_dnn(
    input_dim  = ncol(x_tensor),
    hidden     = hidden,
    activation = activation,
    output_dim = 1L,
    dropout    = dropout,
    batch_norm = batch_norm
  )
  
  net$to(device = device)

  optimizer <- torch::optim_adam(
    net$parameters,
    lr = lr,
    weight_decay = 1e-4
  )

  loss_history <- numeric(epochs)
  early_stopped <- FALSE
  last_epoch_run <- epochs

  for (epoch in 1:epochs) {
    net$train()
    optimizer$zero_grad()

    pred     <- net(x_tensor)
    loss_val <- loss_fn(pred, y_tensor)

    loss_val$backward()
    optimizer$step()

    current_loss <- loss_val$item()
    loss_history[epoch] <- current_loss
    last_epoch_run <- epoch

    if (verbose && epoch %% 50 == 0) {
      cat(sprintf("Epoch %d - Loss: %.6f\n", epoch, current_loss))
      cat("\n")
    }

    ## early stopping on training loss
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

  # truncate history if early stopping
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
      device       = device
    ),
    class = "survdnn"
  )
}
