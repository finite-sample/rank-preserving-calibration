# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

`rank_preserving_calibration` is a Python package for rank-preserving calibration of multiclass probabilities. It implements several algorithms:

1. **Dykstra's alternating projections** (`calibrate_dykstra`) - recommended default method
2. **ADMM optimization** (`calibrate_admm`) - alternative solver with convergence history
3. **KL-divergence calibration** (`calibrate_kl`) - KL-based loss instead of Euclidean
4. **Soft calibration** (`calibrate_soft`) - tunable trade-offs between constraints
5. **Two-stage IPF** (`calibrate_two_stage`) - IPF followed by isotonic projection

The package projects probability matrices onto the intersection of:
- Row-simplex constraints (each row sums to 1, non-negative)
- Isotonic column marginals (values non-decreasing when sorted by original scores, with target column sums)

## Code Architecture

### Core Module Structure
- `rank_preserving_calibration/calibration.py`: Main algorithms and result classes (Dykstra, ADMM)
- `rank_preserving_calibration/kl_calibration.py`: KL-divergence calibration algorithms
- `rank_preserving_calibration/kl_nearly.py`: Nearly isotonic utilities for KL geometry
- `rank_preserving_calibration/soft_calibration.py`: Soft constraint calibration
- `rank_preserving_calibration/two_stage.py`: IPF and two-stage calibration
- `rank_preserving_calibration/ovr_isotonic.py`: One-vs-Rest isotonic regression
- `rank_preserving_calibration/nearly.py`: Nearly isotonic projection utilities
- `rank_preserving_calibration/metrics.py`: Calibration quality metrics
- `rank_preserving_calibration/analysis.py`: Flatness and shift analysis
- `rank_preserving_calibration/__init__.py`: Public API exports and legacy aliases
- `tests/`: Test suite using pytest

### Key Classes and Functions
- `calibrate_dykstra(P, M, **kwargs)`: Main calibration function using Dykstra's method
- `calibrate_admm(P, M, **kwargs)`: Alternative ADMM-based calibration
- `CalibrationResult`: Standard result object with calibrated matrix Q and diagnostics
- `ADMMResult`: ADMM-specific result with convergence history
- `CalibrationError`: Custom exception for invalid inputs

### KL Calibration (kl_calibration.py)
- `calibrate_kl(P, M, R, A)`: KL-divergence calibration with anchor-reference decoupling
- `calibrate_kl_soft(P, M, lam)`: Soft KL calibration with λ-weighted rank penalty
- `calibrate_kl_pareto(P, M, lambda_grid)`: Pareto frontier computation
- `KLCalibrationResult`: Result container for KL calibration
- `KLParetoResult`: Pareto frontier results

### KL Nearly Isotonic (kl_nearly.py)
- `project_near_kl_isotonic(v, eps_slack)`: Multiplicative slack projection for KL
- `prox_kl_near_isotonic(y, lam)`: λ-penalty proximal operator for KL

### Nearly Isotonic Functions (nearly.py)
- `project_near_isotonic_euclidean(v, eps, sum_target=None)`: Epsilon-slack projection
- `prox_near_isotonic(v, lam)`: Lambda-penalty prox operator
- `prox_near_isotonic_with_sum(v, lam, sum_target)`: Prox with sum constraint

### Soft Calibration (soft_calibration.py)
- `calibrate_soft(P, M, lam_m, lam_r)`: Gradient descent with soft penalties
- `calibrate_soft_admm(P, M, lam_m, lam_r, rho)`: ADMM version with better convergence
- `SoftCalibrationResult`: Result dataclass with objective breakdown

### Two-Stage IPF Calibration (two_stage.py)
- `calibrate_ipf(P, M)`: Iterative Proportional Fitting (raking)
- `calibrate_two_stage(P, M)`: IPF followed by isotonic projection
- `IPFResult`, `TwoStageResult`: Result dataclasses

### Flatness & Shift Analysis (analysis.py)
- `flatness_metrics(Q, P, M)`: Measure solution informativeness
- `marginal_shift_metrics(P, M)`: Quantify distribution shift
- `flatness_bound(P, M)`: Theoretical bound on expected flatness
- `compare_calibration_methods(P, M, results)`: Compare multiple methods

### Extended Metrics (metrics.py)
- `column_variance(Q)`: Per-column variance of calibrated probabilities
- `informativeness_ratio(Q, P)`: Compare Q vs P variance (measures flattening)
- `brier(y, probs)`: Brier score for multiclass classification
- `top_label_ece(y, probs)`: Expected calibration error for top predictions
- `classwise_ece(y, probs)`: Per-class calibration error analysis
- `sharpness_metrics(probs)`: Prediction confidence and entropy analysis
- `kl_divergence(Q, R)`: KL divergence between probability matrices

### Algorithm Implementation Details
- Dykstra's method uses alternating projections with memory terms (U, V arrays)
- Row projections use numerically stable simplex projection algorithm
- Column projections use Pool Adjacent Violators (PAV) isotonic regression
- KL calibration uses geometric mean pooling and multiplicative rescaling
- Cycle detection available for Dykstra's method
- ADMM uses augmented Lagrangian with penalty parameter rho

## Development Commands

### Testing
```bash
# Install with test dependencies
uv sync --group test

# Run full test suite
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_basic.py -v

# Run individual test
uv run pytest tests/test_basic.py::TestBasicFunctionality::test_simple_2x2_case -v
```

### Installation and Dependencies
```bash
# Install from source with uv
uv sync

# Install with all optional dependencies
uv sync --all-extras

# Core dependency: numpy>=1.20
# Testing dependencies: pytest>=7.0, pytest-cov, hypothesis
# Example dependencies: scipy>=1.0, matplotlib>=3.0, jupyter, seaborn, scikit-learn
```

### Linting and Formatting
```bash
# Run ruff linter
uv run ruff check .

# Run ruff formatter
uv run ruff format .

# Check formatting without changes
uv run ruff format --check .
```

### Package Building
```bash
# Build wheel and source distribution
uv build

# The package uses uv_build backend with pyproject.toml configuration
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

## Soft Calibration (Addressing Flatness)

When distribution shifts are large, strict constraint satisfaction can produce "flat" solutions with little discrimination between samples. Soft calibration provides tunable trade-offs:

### Soft-Constraint Approach
```python
from rank_preserving_calibration import calibrate_soft

result = calibrate_soft(P, M, lam_m=1.0, lam_r=10.0)
# lam_m: marginal penalty (higher = closer to target M)
# lam_r: rank penalty (higher = more isotonic)
```
- Minimizes: ||Q - P||² + lam_m·||col_sums(Q) - M||² + lam_r·rank_penalty(Q)
- Hard constraint: rows sum to 1 (simplex)
- Allows exploration of Pareto frontier between fit, marginals, and ranks

### Two-Stage IPF Approach
```python
from rank_preserving_calibration import calibrate_two_stage

result = calibrate_two_stage(P, M)
```
- Stage 1: IPF (raking) to match marginals while preserving relative structure
- Stage 2: Isotonic projection to restore rank ordering
- Often produces less flat solutions than direct Dykstra projection

### Diagnosing Flatness
```python
from rank_preserving_calibration import flatness_metrics, marginal_shift_metrics, flatness_bound

shift = marginal_shift_metrics(P, M)  # How big is the shift?
bound = flatness_bound(P, M)  # What flatness should we expect?
flat = flatness_metrics(Q, P, M)  # How flat is the solution?
```

## KL Divergence Calibration

For label-shift scenarios where KL divergence is more natural than Euclidean distance:

### Basic KL Calibration
```python
from rank_preserving_calibration import calibrate_kl

result = calibrate_kl(P, M)
```

### Anchor-Reference Decoupling
```python
# Use P for KL reference, but A for rank ordering
result = calibrate_kl(P, M, R=P, A=A)
```

### Pareto Frontier
```python
from rank_preserving_calibration import calibrate_kl_pareto

result = calibrate_kl_pareto(P, M)
for lam, Q in result.solutions:
    print(f"λ={lam:.2f}: KL={result.kl_values[i]:.4f}")
```

## Nearly Isotonic Calibration

The package supports "nearly isotonic" constraints that allow small violations of strict monotonicity:

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
- **Source**: `docs/source/` contains all Markdown (.md) files (using MyST parser)
- **Configuration**: `docs/source/conf.py` with autodoc, Furo theme, and extensions
- **Build**: `docs/Makefile` and `docs/make.bat` for local building
- **Deployment**: `.github/workflows/docs.yml` automatically builds and deploys to GitHub Pages

### Documentation Commands
```bash
# Install documentation dependencies
uv sync --extra docs

# Build documentation locally
cd docs && make html

# View built docs
open _build/html/index.html
```

### Key Documentation Files
- `index.md`: Main landing page with overview and quick start
- `installation.md`: Setup and dependency instructions
- `quickstart.md`: Practical usage examples
- `theory.md`: Mathematical foundations and algorithms
- `examples.md`: Real-world use cases and scenarios
- `api.md`: Complete API reference with autodoc
- `changelog.md`: Version history

### Documentation URL
**Live documentation**: https://finite-sample.github.io/rank_preserving_calibration/

### Build Artifacts
- Local builds create `docs/_build/` (excluded from git via .gitignore)
- GitHub Actions builds and deploys automatically on push to main
- No need to commit built HTML files

## CI/CD and Quality

The repository uses GitHub Actions for CI with:
- Python 3.12+ testing environment (3.12, 3.13, 3.14)
- Installation via `uv sync --group test`
- Test execution with `uv run pytest tests/ -v`
- Linting with `uv run ruff check .` and `uv run ruff format --check .`
- Automated workflows for both CI testing and releases
- Documentation building and deployment to GitHub Pages
- Sigstore-signed releases

### Local Development
- Use `uv` for dependency management
- Run `uv run ruff format .` before committing
- Run `uv run pytest tests/ -v` to verify changes
- You are on macOS locally
