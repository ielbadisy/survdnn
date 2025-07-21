
## R CMD check results
0 ERRORs, 0 WARNINGs, 1 NOTEs.

- “IBS” and “coxtime” are technical terms referring to integrated Brier score and a time-dependent Cox loss function, respectively.

## Test environments
- Local: Ubuntu 22.04, R 4.4.0
- R-hub: Windows Server 2022 (R-devel), macOS 12 (R-release), Fedora Linux (R-release)
- GitHub Actions CI: ubuntu-latest (R 4.3, 4.4)

## Downstream dependencies
This is a new package, so there are no downstream reverse dependencies.

## Comments


The following issues raised during initial review have been addressed:

- Added appropriate method references in the DESCRIPTION using CRAN-compliant <doi:...> syntax.

- Replaced \\dontrun{} with \\donttest{} in examples and unwrapped examples that run in under 5 seconds (gridsearch_survdnn.Rd).

This is the first CRAN release of `survdnn`, a package for deep neural network-based survival analysis using the `torch` backend in R. It includes support for several common loss functions (Cox, CoxL2 AFT, coxtime), a formula interface, and tools for model evaluation and tuning.

All functions are documented with examples. The package passes all checks and is tested on multiple platforms.

Thank you for reviewing.
