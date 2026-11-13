test_that("predict.survdnn works for type = 'lp'", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  mod <- survdnn(Surv(time, status) ~ age + karno + celltype,
                 data = data, epochs = 10, verbose = FALSE)

  lp <- predict(mod, newdata = data, type = "lp")

  expect_type(lp, "double")
  expect_length(lp, nrow(data))
  expect_false(anyNA(lp))
})

test_that("predict.survdnn works for type = 'survival'", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  times <- c(30, 90, 180)
  mod <- survdnn(Surv(time, status) ~ age + karno + celltype,
                 data = data, epochs = 10, verbose = FALSE)

  surv <- predict(mod, newdata = data, type = "survival", times = times)

  expect_s3_class(surv, "data.frame")
  expect_equal(ncol(surv), length(times))
  expect_equal(nrow(surv), nrow(data))
  expect_true(all(surv >= 0 & surv <= 1))
  expect_named(surv, paste0("t=", times))
})

test_that("predict.survdnn works for type = 'risk'", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  mod <- survdnn(Surv(time, status) ~ age + karno + celltype,
                 data = data, epochs = 10, verbose = FALSE)

  risk <- predict(mod, newdata = data, type = "risk", times = 180)

  expect_type(risk, "double")
  expect_length(risk, nrow(data))
  expect_true(all(risk >= 0 & risk <= 1))
})

test_that("predict.survdnn handles invalid or missing times properly", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  data <- survival::veteran
  mod <- survdnn(Surv(time, status) ~ age + karno + celltype,
                 data = data, epochs = 10, verbose = FALSE)

  expect_error(predict(mod, newdata = data, type = "survival"),
               "`times` must be specified")

  
  expect_error(predict(mod, newdata = data, type = "risk", times = c(30, 60)),
                "For type = 'risk', `times` must be a single( numeric)? value\\.")
              
})

test_that("Cox survival predictions use the Breslow baseline with lp as an offset", {
  skip_on_cran()
  skip_if_not(torch::torch_is_installed())

  set.seed(1)
  n <- 300
  d <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  t_ev <- rexp(n, 0.1 * exp(0.8 * d$x1 - 0.5 * d$x2))
  t_cens <- rexp(n, 0.05)
  d$time <- pmin(t_ev, t_cens)
  d$status <- as.integer(t_ev <= t_cens)

  torch::torch_manual_seed(1)
  mod <- survdnn(Surv(time, status) ~ x1 + x2, data = d,
                 hidden = 8, epochs = 50, verbose = FALSE)

  lp_train <- predict(mod, newdata = d, type = "lp")
  nd <- d[1:5, ]
  lp_new <- predict(mod, newdata = nd, type = "lp")
  times <- c(0.01, 1, 5, 10, 20, 1000)

  pred <- unname(as.matrix(predict(mod, newdata = nd, times = times, type = "survival")))

  fit0 <- survival::coxph(survival::Surv(time, status) ~ offset(lp),
                          data = data.frame(time = d$time, status = d$status, lp = lp_train),
                          ties = "breslow")
  s0 <- summary(survival::survfit(fit0), times = times, extend = TRUE)$surv
  ref <- t(vapply(lp_new, function(l) s0^exp(l - mean(lp_train)), numeric(length(times))))

  expect_equal(pred, ref, tolerance = 1e-6)
})
