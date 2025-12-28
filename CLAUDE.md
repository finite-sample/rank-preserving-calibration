# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

`rank_preserving_calibration` is a Python package for rank-preserving calibration of multiclass probabilities. It implements two main algorithms:

1. **Dykstra's alternating projections** (`calibrate_dykstra`) - recommended default method
2. **ADMM optimization** (`calibrate_admm`) - alternative solver with convergence history

The package projects probability matrices onto the intersection of:
- Row-simplex constraints (each row sums to 1, non-negative)
- Isotonic column marginals (values non-decreasing when sorted by original scores, with target column sums)

## Code Architecture

### Core Module Structure
- `rank_preserving_calibration/calibration.py`: Main algorithms and result classes (Dykstra, ADMM)
- `rank_preserving_calibration/ovr_isotonic.py`: One-vs-Rest isotonic regression
- `rank_preserving_calibration/nearly.py`: Nearly isotonic projection utilities
- `rank_preserving_calibration/__init__.py`: Public API exports and legacy aliases
- `examples/data_helpers.py`: Synthetic data generation utilities (not part of main package)
- `tests/`: Test suite using pytest

### Key Classes and Functions
- `calibrate_dykstra(P, M, **kwargs)`: Main calibration function using Dykstra's method
- `calibrate_admm(P, M, **kwargs)`: Alternative ADMM-based calibration
- `CalibrationResult`: Standard result object with calibrated matrix Q and diagnostics
- `ADMMResult`: ADMM-specific result with convergence history
- `CalibrationError`: Custom exception for invalid inputs

### Nearly Isotonic Functions (New)
- `project_near_isotonic_euclidean(v, eps, sum_target=None)`: Epsilon-slack projection
- `prox_near_isotonic(v, lam)`: Lambda-penalty prox operator
- `prox_near_isotonic_with_sum(v, lam, sum_target)`: Prox with sum constraint

### Algorithm Implementation Details
- Dykstra's method uses alternating projections with memory terms (U, V arrays)
- Row projections use numerically stable simplex projection algorithm
- Column projections use Pool Adjacent Violators (PAV) isotonic regression
- Cycle detection available for Dykstra's method
- ADMM uses augmented Lagrangian with penalty parameter rho

## Development Commands

### Testing
```bash
# Install with test dependencies
pip install -e ".[testing]"

# Run full test suite
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_basic.py -v

# Run individual test
python -m pytest tests/test_basic.py::TestBasicFunctionality::test_simple_2x2_case -v
```

### Installation and Dependencies
```bash
# Install from source
pip install .

# Install with all optional dependencies (examples, testing)
pip install -e ".[all]"

# Core dependency: numpy>=1.18
# Testing dependencies: pytest>=6.0, pytest-cov
# Example dependencies: scipy>=1.0, matplotlib>=3.0, jupyter, seaborn
```

### Package Building
```bash
# Build wheel and source distribution
python -m build

# The package uses setuptools with pyproject.toml configuration
# No additional build tools (Makefile, tox, etc.) are configured
```

## Working with Examples

The package provides comprehensive documentation examples at `/docs/source/examples/`:
- Medical diagnosis calibration using breast cancer dataset
- Financial risk assessment with credit scoring
- Text classification with sentiment analysis
- Computer vision with handwritten digits
- Survey research with demographic reweighting

Testing utilities are located in `tests/data_helpers.py`:
- `create_test_case()`: Generate synthetic test data for various scenarios
- `create_realistic_classifier_case()`: Simulate miscalibrated classifiers
- `create_survey_reweighting_case()`: Generate survey bias scenarios
- `analyze_calibration_result()`: Analyze calibration impact

Test data scenarios support:
- `"random"`: Dirichlet-generated probabilities
- `"skewed"`: Biased class distributions
- `"linear"`: Linear trends for rank preservation testing
- `"challenging"`: Difficult cases with potential feasibility issues

## Nearly Isotonic Calibration (New Feature)

The package now supports "nearly isotonic" constraints that allow small violations of strict monotonicity:

### Epsilon-Slack Approach (Dykstra)
```python
nearly_params = {"mode": "epsilon", "eps": 0.05}
result = calibrate_dykstra(P, M, nearly=nearly_params)
```
- Allows z[i+1] >= z[i] - eps instead of strict z[i+1] >= z[i]
- Uses Euclidean projection onto convex slack constraint set
- Maintains convergence guarantees of Dykstra's method

### Lambda-Penalty Approach (ADMM)
```python
nearly_params = {"mode": "lambda", "lam": 1.0}
result = calibrate_admm(P, M, nearly=nearly_params)
```
- Penalizes isotonicity violations with λ * sum(max(0, z[i] - z[i+1]))
- Uses proximal operator for soft isotonic constraint
- Experimental - may require parameter tuning

## Important Implementation Notes

### Numerical Stability
- All computations use float64 precision
- Isotonic regression includes fallback for numerical edge cases
- Input validation checks for NaN/infinite values and negative probabilities
- Feasibility warnings when sum(M) differs significantly from N

### Nearly Isotonic Notes
- Epsilon-slack maintains convexity and theoretical guarantees
- Lambda-penalty approach is more experimental and may need tuning
- Both approaches can be less restrictive than strict isotonic constraints


### Common Parameter Patterns
- `P`: Input probability matrix (N×J)
- `M`: Target column sums (length J)
- `max_iters`: Algorithm iteration limits (3000 for Dykstra, 1000 for ADMM)
- `tol`: Convergence tolerances (1e-7 for Dykstra, 1e-6 for ADMM)
- `verbose`: Progress printing flag
- `rtol`: Relative tolerance for isotonic regression (1e-12)
- `nearly`: Dict with "mode" ("epsilon" or "lambda") and parameters

## Documentation

The repository has comprehensive Sphinx documentation deployed to GitHub Pages:

### Documentation Structure
- **Source**: `docs/source/` contains all reStructuredText (.rst) files
- **Configuration**: `docs/source/conf.py` with autodoc, RTD theme, and extensions
- **Build**: `docs/Makefile` and `docs/make.bat` for local building
- **Deployment**: `.github/workflows/docs.yml` automatically builds and deploys to GitHub Pages

### Documentation Commands
```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation locally
cd docs && make html

# View built docs
open _build/html/index.html
```

### Key Documentation Files
- `index.rst`: Main landing page with overview and quick start
- `installation.rst`: Setup and dependency instructions
- `quickstart.rst`: Practical usage examples
- `theory.rst`: Mathematical foundations and algorithms
- `examples.rst`: Real-world use cases and scenarios
- `api.rst`: Complete API reference with autodoc
- `changelog.rst`: Version history

### Documentation URL
**Live documentation**: https://finite-sample.github.io/rank_preserving_calibration/

### Build Artifacts
- Local builds create `docs/_build/` (excluded from git via .gitignore)
- GitHub Actions builds and deploys automatically on push to main
- No need to commit built HTML files

## CI/CD and Quality

The repository uses GitHub Actions for CI with:
- Python 3.11+ testing environment
- Installation via `pip install -e ".[testing]"`
- Test execution with `python -m pytest tests/ -v`
- Automated workflows for both CI testing and releases
- Documentation building and deployment to GitHub Pages

No additional linting, formatting, or coverage tools are configured in the current setup.
- you are on mac os locally
