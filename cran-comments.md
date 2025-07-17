
## R CMD check results
There were no ERRORs, WARNINGs, or NOTEs.

## Test environments
- Local: Ubuntu 22.04, R 4.4.0
- R-hub: Windows Server 2022 (R-devel), macOS 12 (R-release), Fedora Linux (R-release)
- GitHub Actions CI: ubuntu-latest (R 4.3, 4.4)

## Downstream dependencies
This is a new package, so there are no downstream reverse dependencies.

## Comments
This is the first CRAN release of `survdnn`, a package for deep neural network-based survival analysis using the `torch` backend in R. It includes support for several common loss functions (Cox, CoxL2 AFT, coxtime), a formula interface, and tools for model evaluation and tuning.

All functions are documented with examples. The package passes all checks and is tested on multiple platforms.

Thank you for reviewing.
