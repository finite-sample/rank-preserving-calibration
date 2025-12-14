# Changelog

## Version 0.6.0 (2024-10-XX)

**Breaking Changes:**
* **Python 3.11+ Required**: Dropped support for Python 3.9 and 3.10
* Modernized type annotations using built-in types (`tuple[]` instead of `Tuple[]`)

**Modernization:**
* Removed backward compatibility imports (`tomli`, `importlib_metadata`)
* Updated to use Python 3.11+ standard library features (`tomllib`, `importlib.metadata`)
* Modernized CI/CD matrix: Python 3.11 on all platforms, 3.12/3.13 on Linux only
* Updated Furo documentation theme (replacing sphinx_rtd_theme)

**Documentation:**
* **New**: Comprehensive metrics module documentation in README and autodocs
* **New**: Complete evaluation workflow examples with all 11 metrics functions
* **New**: Detailed metrics coverage including feasibility_metrics, isotonic_metrics, distance_metrics, scoring functions (NLL, Brier, ECE), sharpness_metrics, and auc_deltas
* **New**: Metrics usage table with function purposes and complete evaluation workflow code examples
* Added metrics demonstration to focused_nearly_isotonic_example.py with comprehensive comparison
* Updated Sphinx autodocs to include metrics module with full API documentation
* Updated all Python version references to reflect 3.11+ requirement

**Infrastructure:**
* GitHub Actions workflow now supports environment selection for manual publishing
* Updated publish workflow to test only supported Python versions (3.11-3.13)
* Improved ruff configuration with Greek letter support (ρ, λ) for mathematical notation

**Quality:**
* All code passes latest ruff linting with Python 3.11 target
* Consistent code formatting across entire codebase
* Removed unused imports and modernized import patterns

## Version 0.4.1 (2024-08-XX)

**Bug Fixes:**
* Minor PyPI release improvements

## Version 0.4.0 (2024-08-XX)

**New Features:**
* Added nearly isotonic calibration with epsilon-slack and lambda-penalty approaches
* New functions: `project_near_isotonic_euclidean`, `prox_near_isotonic`, `prox_near_isotonic_with_sum`
* Enhanced API with `nearly` parameter for both Dykstra and ADMM methods

**API Changes:**
* Improved result classes with better diagnostics
* Enhanced error handling and input validation

**Documentation:**
* Added comprehensive Sphinx documentation
* Improved examples and tutorials
* Added mathematical theory section

## Version 0.3.x and Earlier

**Core Features:**
* Implementation of Dykstra's alternating projections algorithm
* ADMM-based optimization with convergence tracking
* Row-simplex and isotonic column constraints
* Robust numerical implementation with PAV algorithm
* Comprehensive test suite