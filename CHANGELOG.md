# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.1] - 2024-12-21

### Changed
- **Removed tqdm dependency**
  - Eliminated optional import patterns for cleaner dependency management
  - Removed `progress_bar` parameter from `calibrate_dykstra` and `calibrate_admm` functions
  - Updated examples and documentation to remove progress bar references

- **Enhanced docstrings**
  - Converted all docstrings to Google style format for consistency
  - Added comprehensive examples, args, and returns documentation
  - Improved API documentation across all public functions

- **Python modernization**
  - Refactored input validation to use Python 3.11+ match statements
  - Updated validation patterns for better readability and maintainability

### Removed
- **Breaking Change**: `progress_bar` parameter no longer supported
- Removed tqdm from both performance and dev dependency groups

## [0.7.0] - 2024-12-13

### Added
- **Performance optimizations with Numba JIT compilation**
  - JIT-compiled versions of performance-critical functions (`project_row_simplex`, `isotonic_regression`)
  - 1.5-3x speedup on moderate-sized problems when Numba is available
  - Graceful fallback to pure Python when Numba is not installed
  - New `use_jit` parameter for `calibrate_dykstra` and `calibrate_admm` (defaults to True)

- **Progress bar support for long-running calibrations**
  - Optional tqdm progress bars via `progress_bar` parameter
  - Works with both Dykstra and ADMM algorithms
  - Graceful fallback when tqdm is not installed

- **Modern Python 3.11+ features**
  - Added `slots=True` to dataclasses for better memory efficiency
  - Use pathlib instead of os.path for file operations
  - Updated to use importlib.metadata for version handling

### Changed
- **Dependency management improvements**
  - Added optional performance dependencies: `numba>=0.56`, `tqdm>=4.65`
  - Configured deptry for proper dependency tracking
  - Updated dev dependency groups to include performance deps

- **Code quality enhancements**
  - Fixed all linting issues for consistent code formatting
  - Improved numerical tolerance handling in tests
  - Enhanced error messages and validation

### Fixed
- Improved numerical stability in edge cases
- Better handling of optional dependencies with proper fallbacks
- Fixed test tolerances to account for expected JIT floating-point differences

## [0.6.0] - Previous Release
- Basic rank-preserving calibration functionality
- Dykstra's alternating projections algorithm
- ADMM optimization algorithm
- Nearly isotonic constraints (epsilon-slack and lambda-penalty)
- Comprehensive test suite and documentation
