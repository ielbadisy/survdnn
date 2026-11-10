## R CMD check results

0 errors | 0 warnings | 0 notes

## Update

This is an update from version 1.0.0 to 1.0.2.

Changes made in this update:

* Dropped the `dplyr`, `purrr`, `tidyr`, `tibble` and `tidyselect`
  dependencies; data handling now uses `basetable` and base R.
* Fixed the `gridsearch_survdnn()` example.
* `DESCRIPTION` now cites only the package's R Journal paper
  (<doi:10.32614/RJ-2026-008>).
