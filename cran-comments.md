## R CMD check results

0 errors | 0 warnings | 0 notes

## Resubmission

This is a resubmission for version 0.7.6.

Changes made in this update:

* Guarded all model-training `\\donttest{}` examples with `torch::torch_is_installed()` so `--run-donttest` succeeds on systems without Torch installed.
* Updated examples to use explicit dataset access via `survival::veteran`.
* Added explicit `verbose` arguments to `cv_survdnn()` and `tune_survdnn()`, and improved consistency of progress messages across fit/cv/tune.
* Regenerated documentation (`man/*.Rd`) after example updates.
* Removed internal-tool references from package notes.
