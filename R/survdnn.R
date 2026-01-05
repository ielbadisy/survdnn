#' Build a Deep Neural Network for Survival Analysis
#'
#' Constructs a multilayer perceptron (MLP) with optional batch normalization
#' and dropout. Used internally by [survdnn()] to define the model architecture.
#'
#' @param input_dim Integer. Number of input features.
#' @param hidden Integer vector. Sizes of the hidden layers (e.g., c(32, 16)).
#' @param activation Character. Name of the activation function to use in each layer.
#'   Supported options: `"relu"`, `"leaky_relu"`, `"tanh"`, `"sigmoid"`, `"gelu"`,
#'   `"elu"`, `"softplus"`.
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
#' Fits a deep neural network (MLP) for right-censored time-to-event data using
#' one of the supported losses: Cox partial likelihood, L2-penalized Cox,
#' log-normal AFT (censored negative log-likelihood), or CoxTime (time-dependent
#' relative risk model).
#'
#' The function:
#' \itemize{
#'   \item builds an MLP via [build_dnn()],
#'   \item preprocesses predictors using centering/scaling (stored in the model),
#'   \item optionally applies log-time centering for AFT (stored as \code{aft_loc}),
#'   \item scales time for CoxTime to stabilize optimization (stored as \code{coxtime_time_center}/\code{coxtime_time_scale}),
#'   \item trains the network with a torch optimizer and optional callbacks.
#' }
#'
#' @param formula A survival formula of the form \code{Surv(time, status) ~ predictors}.
#' @param data A data frame containing the variables in the model.
#' @param hidden Integer vector giving hidden layer widths (e.g., \code{c(32L, 16L)}).
#' @param activation Activation function used in each hidden layer. One of
#'   \code{"relu"}, \code{"leaky_relu"}, \code{"tanh"}, \code{"sigmoid"},
#'   \code{"gelu"}, \code{"elu"}, \code{"softplus"}.
#' @param lr Learning rate passed to the optimizer (default \code{1e-4}).
#' @param epochs Number of training epochs (default \code{300L}).
#' @param loss Loss function to optimize. One of \code{"cox"}, \code{"cox_l2"},
#'   \code{"aft"}, \code{"coxtime"}.
#' @param optimizer Optimizer name. One of \code{"adam"}, \code{"adamw"},
#'   \code{"sgd"}, \code{"rmsprop"}, \code{"adagrad"}.
#' @param optim_args Optional named list of extra arguments passed to the chosen
#'   torch optimizer (e.g., \code{list(weight_decay = 1e-4, momentum = 0.9)}).
#' @param verbose Logical; whether to print training progress every 50 epochs.
#' @param dropout Dropout rate applied after each hidden layer (set \code{0} to disable).
#' @param batch_norm Logical; whether to add batch normalization after each hidden linear layer.
#' @param callbacks Optional callback(s) for early stopping or monitoring.
#'   May be \code{NULL}, a single function, or a list of functions. Each callback must have
#'   signature \code{function(epoch, current_loss)} and return \code{TRUE} to stop training,
#'   \code{FALSE} otherwise.
#' @param .seed Optional integer seed controlling both R and torch RNGs (weight init,
#'   shuffling, dropout) for reproducibility.
#' @param .device Computation device. One of \code{"auto"}, \code{"cpu"}, \code{"cuda"}.
#'   \code{"auto"} selects CUDA when available.
#' @param na_action Missing-data handling. \code{"omit"} drops incomplete rows (and reports
#'   how many were removed when \code{verbose=TRUE}); \code{"fail"} errors if any missing
#'   values are present in model variables.
#'
#' @details
#' \strong{AFT model.} With \code{loss="aft"}, the model is a log-normal AFT model:
#' \deqn{\log(T) = \text{aft\_loc} + \mu_{\text{resid}}(x) + \sigma \varepsilon, \quad \varepsilon \sim \mathcal{N}(0,1).}
#' For numerical stability, training uses centered log-times
#' \code{log(time) - aft_loc}. The learned network output corresponds to
#' \code{mu_resid(x)}. The fitted object stores \code{aft_loc} and the learned global
#' \code{aft_log_sigma}.
#'
#' \strong{CoxTime.} With \code{loss="coxtime"}, the network represents a time-dependent
#' score \eqn{g(t, x)}. Internally, time is standardized before being concatenated with
#' standardized covariates. The scaling parameters are stored as
#' \code{coxtime_time_center} and \code{coxtime_time_scale} to ensure prediction uses the
#' same transformation.
#'
#' @return An object of class \code{"survdnn"} with components:
#' \describe{
#'   \item{model}{Trained torch \code{nn_module} (MLP).}
#'   \item{formula}{Model formula used for fitting.}
#'   \item{data}{Training data used for fitting (original \code{data} argument).}
#'   \item{xnames}{Predictor column names used by the model matrix.}
#'   \item{x_center}{Numeric vector of predictor means used for scaling.}
#'   \item{x_scale}{Numeric vector of predictor standard deviations used for scaling.}
#'   \item{loss_history}{Numeric vector of loss values per epoch (possibly truncated by early stopping).}
#'   \item{final_loss}{Final loss value (last element of \code{loss_history}).}
#'   \item{loss}{Loss name used for training.}
#'   \item{activation}{Activation function name.}
#'   \item{hidden}{Hidden layer sizes.}
#'   \item{lr}{Learning rate.}
#'   \item{epochs}{Number of requested epochs.}
#'   \item{optimizer}{Optimizer name.}
#'   \item{optim_args}{List of optimizer arguments used.}
#'   \item{device}{Torch device used for fitting.}
#'   \item{dropout}{Dropout rate used.}
#'   \item{batch_norm}{Whether batch normalization was used.}
#'   \item{na_action}{Missing-data strategy used.}
#'   \item{aft_log_sigma}{Learned global \code{log(sigma)} for AFT; \code{NA_real_} otherwise.}
#'   \item{aft_loc}{Log-time centering offset used for AFT; \code{NA_real_} otherwise.}
#'   \item{coxtime_time_center}{Time centering used for CoxTime; \code{NA_real_} otherwise.}
#'   \item{coxtime_time_scale}{Time scaling used for CoxTime; \code{NA_real_} otherwise.}
#' }
#'
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
#'   veteran <- survival::veteran
#'
#'   # --- Cox model ---
#'   fit_cox <- survdnn(
#'     Surv(time, status) ~ age + karno + celltype,
#'     data = veteran,
#'     epochs = 50,
#'     verbose = FALSE,
#'     .seed = 1
#'   )
#'   lp <- predict(fit_cox, newdata = veteran, type = "lp")
#'   S  <- predict(fit_cox, newdata = veteran, type = "survival", times = c(30, 90, 180))
#'
#'   # --- AFT log-normal model ---
#'   fit_aft <- survdnn(
#'     Surv(time, status) ~ age + karno + celltype,
#'     data = veteran,
#'     loss = "aft",
#'     epochs = 50,
#'     verbose = FALSE,
#'     .seed = 1
#'   )
#'   S_aft <- predict(fit_aft, newdata = veteran, type = "survival", times = c(30, 90, 180))
#'
#'   # --- CoxTime model ---
#'   fit_ct <- survdnn(
#'     Surv(time, status) ~ age + karno + celltype,
#'     data = veteran,
#'     loss = "coxtime",
#'     epochs = 50,
#'     verbose = FALSE,
#'     .seed = 1
#'   )
#'   # By default, CoxTime survival predictions can use event times if times=NULL
#'   S_ct <- predict(fit_ct, newdata = veteran, type = "survival")
#' }
#' }
#'
#' @export

survdnn <- function(formula,
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
                    na_action = c("omit", "fail")) {

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
    } else if (!is.list(callbacks) ||
               !all(vapply(callbacks, is.function, logical(1)))) {
      stop("`callbacks` must be NULL, a function, or a list of functions.",
           call. = FALSE)
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

  n_after   <- nrow(mf)
  n_removed <- n_before - n_after ## keep it informative 

  if (n_removed > 0 && isTRUE(verbose) && na_action == "omit") {
    message(sprintf("Removed %d observations with missing values.", n_removed))
  }

  y        <- model.response(mf)
  x        <- model.matrix(attr(mf, "terms"), data = mf)[, -1, drop = FALSE]
  time     <- y[, "time"]
  status   <- y[, "status"]
  x_scaled <- scale(x)

  # AFT location offset
  aft_loc <- NA_real_

  if (loss == "aft") {
    evt <- status == 1
    aft_loc <- if (any(evt)) {
      mean(log(pmax(time[evt], .Machine$double.eps)))
    } else {
      mean(log(pmax(time, .Machine$double.eps)))
    }

    if (!is.finite(aft_loc)) aft_loc <- 0
  }

  # CoxTime scaling
  coxtime_time_center <- NA_real_
  coxtime_time_scale  <- NA_real_
  time_scaled <- NULL

  if (loss == "coxtime") {
    ts <- scale(as.numeric(time))
    coxtime_time_center <- as.numeric(attr(ts, "scaled:center"))
    coxtime_time_scale  <- as.numeric(attr(ts, "scaled:scale"))

    if (!is.finite(coxtime_time_scale) || coxtime_time_scale <= 0) {
      coxtime_time_scale <- 1
    }

    time_scaled <- as.numeric(ts)
  }

  # tensors
  x_tensor <- if (loss == "coxtime") { ## special case only for coxtime since it needs time_scaled with x_scaled
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

  y_tensor <- torch::torch_tensor(
    cbind(time, status),
    dtype  = torch::torch_float(),
    device = device
  )

  # network building
  net <- build_dnn(
    input_dim  = ncol(x_tensor),
    hidden     = hidden,
    activation = activation,
    output_dim = 1L,
    dropout    = dropout,
    batch_norm = batch_norm
  )
  net$to(device = device)

  # Loss dispatcher; AFT initializes a learnable log(sigma) and CoxTime uses a custom factory

  extra_params  <- NULL
  aft_log_sigma <- NA_real_

  loss_fn <- switch(
    loss,
    cox = function(net, x, y) cox_loss(net(x), y),
    cox_l2 = function(net, x, y) cox_l2_loss(net(x), y, lambda = 1e-3),
    aft = {
      aft_bundle <- survdnn__aft_lognormal_nll_factory(
        device  = device,
        aft_loc = aft_loc
      )
      extra_params <<- aft_bundle$extra_params
      function(net, x, y) aft_bundle$loss_fn(net, x, y)
    },
    coxtime = {
      lf <- survdnn__coxtime_loss_factory(net)
      function(net, x, y) lf(x, y)
    }
  )

  # optimizer
  params <- net$parameters

  if (loss == "aft" && !is.null(extra_params$log_sigma)) {
    params <- c(params, list(extra_params$log_sigma))
  }

  opt_args <- c(list(params = params, lr = lr), optim_args)

  if (is.null(optim_args$weight_decay) &&
      optimizer %in% c("adam", "adamw")) {
    opt_args$weight_decay <- 1e-4
  }

  optimizer_obj <- switch(
    optimizer,
    adam    = do.call(torch::optim_adam,    opt_args),
    adamw   = do.call(torch::optim_adamw,   opt_args),
    sgd     = do.call(torch::optim_sgd,     opt_args),
    rmsprop = do.call(torch::optim_rmsprop, opt_args),
    adagrad = do.call(torch::optim_adagrad, opt_args)
  )

  # training loop 
  loss_history   <- numeric(epochs)
  early_stopped  <- FALSE
  last_epoch_run <- epochs
  
  ##the the training is purely synchronous so no need to use coro
  for (epoch in seq_len(epochs)) { 
    net$train()
    optimizer_obj$zero_grad()

    loss_val <- loss_fn(net, x_tensor, y_tensor)
    loss_val$backward()
    optimizer_obj$step()

    current_loss <- loss_val$item()
    loss_history[epoch] <- current_loss
    last_epoch_run <- epoch

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

  if (loss == "aft" && !is.null(extra_params$log_sigma)) {
    aft_log_sigma <- as.numeric(extra_params$log_sigma$item())
    if (!is.finite(aft_log_sigma)) aft_log_sigma <- NA_real_
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
