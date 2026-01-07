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
build_dnn <- function(
  input_dim,
  hidden,
  activation = "relu",
  output_dim = 1L,
  dropout = 0.3,
  batch_norm = TRUE
) {
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
#' Trains a deep neural network (DNN) to model right-censored survival data
#' using one of the predefined loss functions: Cox, AFT, or Coxtime.
#'
#' @param formula A survival formula of the form `Surv(time, status) ~ predictors`.
#' @param data A data frame containing the variables in the model.
#' @param hidden Integer vector. Sizes of the hidden layers (default: c(32, 16)).
#' @param activation Character string specifying the activation function to use in each layer.
#'   Supported options: `"relu"`, `"leaky_relu"`, `"tanh"`, `"sigmoid"`, `"gelu"`, `"elu"`, `"softplus"`.
#' @param lr Learning rate for the optimizer (default: `1e-4`).
#' @param epochs Number of training epochs (default: 300).
#' @param loss Character name of the loss function to use. One of `"cox"`, `"cox_l2"`, `"aft"`, or `"coxtime"`.
#' @param optimizer Character string specifying the optimizer to use. One of
#'   `"adam"`, `"adamw"`, `"sgd"`, `"rmsprop"`, or `"adagrad"`. Defaults to `"adam"`.
#' @param optim_args Optional named list of additional arguments passed to the
#'   underlying torch optimizer (e.g., `list(weight_decay = 1e-4, momentum = 0.9)`).
#' @param verbose Logical; whether to print loss progress every 50 epochs (default: TRUE).
#' @param dropout Numeric between 0 and 1. Dropout rate applied after each
#'   hidden layer (default = 0.3). Set to 0 to disable dropout.
#' @param batch_norm Logical; whether to add batch normalization after each
#'   hidden linear layer (default = TRUE).
#' @param callbacks Optional list of callback functions. Each callback should have
#'   signature `function(epoch, current)` and return TRUE if training should stop,
#'   FALSE otherwise. Used, for example, with [callback_early_stopping()].
#' @param .seed Optional integer. If provided, sets both R and torch random seeds for reproducible
#'   weight initialization, shuffling, and dropout.
#' @param .device Character string indicating the computation device.
#'   One of `"auto"`, `"cpu"`, or `"cuda"`. `"auto"` uses CUDA if available,
#'   otherwise falls back to CPU.
#' @param na_action Character. How to handle missing values in the model variables:
#'   `"omit"` drops incomplete rows (and reports how many were removed when `verbose=TRUE`);
#'   `"fail"` stops with an error if any missing values are present.
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
#'   \item{optimizer}{Optimizer name used.}
#'   \item{optim_args}{List of optimizer arguments used.}
#'   \item{device}{Torch device used for training (`torch_device`).}
#'   \item{aft_log_sigma}{Learned global log(sigma) for `loss="aft"`; `NA_real_` otherwise.}
#'   \item{aft_loc}{AFT log-time location offset used for centering when `loss="aft"`; `NA_real_` otherwise.}
#'   \item{coxtime_time_center}{Mean used to scale time for CoxTime; `NA_real_` otherwise.}
#'   \item{coxtime_time_scale}{SD used to scale time for CoxTime; `NA_real_` otherwise.}
#' }
#' @export
survdnn <- function(
  formula,
  data,
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
  .device = c("auto", "cpu", "cuda"),
  na_action = c("omit", "fail")
) {
  survdnn_set_seed(.seed)
  device <- survdnn_get_device(.device)

  loss      <- match.arg(loss)
  optimizer <- match.arg(optimizer)
  na_action <- match.arg(na_action)

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

  environment(formula) <- list2env(
    list(Surv = survival::Surv),
    parent = environment(formula)
  )

  # missing data handling
  n_before <- nrow(data)
  mf <- model.frame(
    formula,
    data = data,
    na.action = if (na_action == "omit") stats::na.omit else stats::na.fail
  )
  n_after <- nrow(mf)
  n_removed <- n_before - n_after
  if (n_removed > 0 && isTRUE(verbose) && na_action == "omit") {
    message(sprintf("Removed %d observations with missing values.", n_removed))
  }

  y        <- model.response(mf)
  x        <- model.matrix(attr(mf, "terms"), data = mf)[, -1, drop = FALSE]
  time     <- y[, "time"]
  status   <- y[, "status"]
  x_scaled <- scale(x)

  # AFT location offset for stability
  aft_loc <- NA_real_
  if (loss == "aft") {
    evt <- (status == 1)
    if (any(evt)) {
      aft_loc <- mean(log(pmax(time[evt], .Machine$double.eps)))
    } else {
      aft_loc <- mean(log(pmax(time, .Machine$double.eps)))
    }
    if (!is.finite(aft_loc)) aft_loc <- 0
  }

  # CoxTime time scaling
  coxtime_time_center <- NA_real_
  coxtime_time_scale  <- NA_real_
  time_scaled <- NULL

  if (loss == "coxtime") {
    ts <- scale(as.numeric(time))
    coxtime_time_center <- as.numeric(attr(ts, "scaled:center"))
    coxtime_time_scale  <- as.numeric(attr(ts, "scaled:scale"))
    if (!is.finite(coxtime_time_scale) || coxtime_time_scale <= 0) coxtime_time_scale <- 1
    time_scaled <- as.numeric(ts)
  }

  # x_tensor:
  # - only for coxtime: [time_scaled, x_scaled]  (time as fed to net)
  # - others : [x_scaled]
  x_tensor <- if (loss == "coxtime") {
    torch::torch_tensor(
      cbind(time_scaled, x_scaled),
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

  # y_tensor always uses RAW time for ordering/risk sets
  y_tensor <- torch::torch_tensor(
    cbind(time, status),
    dtype  = torch::torch_float(),
    device = device
  )

  # network
  net <- build_dnn(
    input_dim  = ncol(x_tensor),
    hidden     = hidden,
    activation = activation,
    output_dim = 1L,
    dropout    = dropout,
    batch_norm = batch_norm
  )
  net$to(device = device)

  # loss dispatcher + (optional) AFT extra params
  extra_params  <- NULL # list for AFT, NULL otherwise
  aft_log_sigma <- NA_real_     
  loss_fn <- NULL

  if (loss == "cox") {
    loss_fn <- function(net, x, y) cox_loss(net(x), y)
  } else if (loss == "cox_l2") {
    loss_fn <- function(net, x, y) cox_l2_loss(net(x), y, lambda = 1e-3)
  } else if (loss == "aft") {
    loc0 <- if (is.finite(aft_loc)) aft_loc else 0
    aft_bundle <- survdnn__aft_lognormal_nll_factory(device = device, aft_loc = loc0)
    extra_params <- aft_bundle$extra_params
    loss_fn <- function(net, x, y) aft_bundle$loss_fn(net, x, y)
  } else if (loss == "coxtime") {
    lf <- survdnn__coxtime_loss_factory(net)
    loss_fn <- function(net, x, y) lf(x, y)
  } else {
    stop("Unsupported loss: ", loss, call. = FALSE)
  }

  # optimizer params
  params <- net$parameters
  if (loss == "aft" && !is.null(extra_params) && !is.null(extra_params$log_sigma)) {
    params <- c(params, list(extra_params$log_sigma))
  }

  opt_args <- c(list(params = params, lr = lr), optim_args)

  if (is.null(optim_args$weight_decay) && optimizer %in% c("adam", "adamw")) {
    opt_args$weight_decay <- 1e-4
  }

  optimizer_obj <- switch(
    optimizer,
    adam    = do.call(torch::optim_adam,    opt_args),
    adamw   = do.call(torch::optim_adamw,   opt_args),
    sgd     = do.call(torch::optim_sgd,     opt_args),
    rmsprop = do.call(torch::optim_rmsprop, opt_args),
    adagrad = do.call(torch::optim_adagrad, opt_args),
    stop("Unsupported optimizer: ", optimizer)
  )

  # training loop
  loss_history   <- numeric(epochs)
  early_stopped  <- FALSE
  last_epoch_run <- epochs

  for (epoch in 1:epochs) {
    net$train()
    optimizer_obj$zero_grad()

    loss_val <- loss_fn(net, x_tensor, y_tensor)
    loss_val$backward()
    optimizer_obj$step()

    current_loss        <- loss_val$item()
    loss_history[epoch] <- current_loss
    last_epoch_run      <- epoch

    if (verbose && epoch %% 50 == 0) {
      cat(sprintf("Epoch %d - Loss: %.6f\n\n", epoch, current_loss))
    }

    if (!is.null(callbacks)) {
      for (cb in callbacks) {
        if (isTRUE(cb(epoch, current_loss))) {
          early_stopped <- TRUE
          break
        }
      }
      if (early_stopped) break
    }
  }

  if (early_stopped && last_epoch_run < epochs) {
    loss_history <- loss_history[seq_len(last_epoch_run)]
  }

  # store learned AFT log(sigma) robustly
  if (loss == "aft" && !is.null(extra_params) && !is.null(extra_params$log_sigma)) {
    aft_log_sigma <- as.numeric(extra_params$log_sigma$item())
    if (!is.finite(aft_log_sigma)) aft_log_sigma <- NA_real_
  } else {
    aft_log_sigma <- NA_real_
  }

  structure(
    list(
      model               = net,
      formula             = formula,
      data                = data,
      xnames              = colnames(x),
      x_center            = attr(x_scaled, "scaled:center"),
      x_scale             = attr(x_scaled, "scaled:scale"),
      loss_history        = loss_history,
      final_loss          = tail(loss_history, 1),
      loss                = loss,
      activation          = activation,
      hidden              = hidden,
      lr                  = lr,
      epochs              = epochs,
      optimizer           = optimizer,
      optim_args          = optim_args,
      device              = device,
      dropout             = dropout,
      batch_norm          = batch_norm,
      na_action           = na_action,
      aft_log_sigma       = aft_log_sigma,
      aft_loc             = if (loss == "aft") aft_loc else NA_real_,
      coxtime_time_center = if (loss == "coxtime") coxtime_time_center else NA_real_,
      coxtime_time_scale  = if (loss == "coxtime") coxtime_time_scale  else NA_real_
    ),
    class = "survdnn"
  )
}
