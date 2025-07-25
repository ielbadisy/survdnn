% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tune_survdnn.R
\name{summarize_tune_survdnn}
\alias{summarize_tune_survdnn}
\title{Summarize survdnn Tuning Results}
\usage{
summarize_tune_survdnn(tuning_results, by_time = TRUE)
}
\arguments{
\item{tuning_results}{The full tibble returned by `tune_survdnn(..., return = "all")`.}

\item{by_time}{Logical; whether to group and summarize separately by time points.}
}
\value{
A summarized tibble with mean and standard deviation of performance metrics.
}
\description{
Aggregates cross-validation results from `tune_survdnn(return = "all")`
by configuration, metric, and optionally by time point.
}
