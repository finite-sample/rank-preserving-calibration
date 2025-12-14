# Quick Start Guide

This guide will get you started with rank-preserving calibration in just a few minutes.

## Basic Usage

The most common use case is calibrating multiclass probabilities using Dykstra's method:

```python
import numpy as np
from rank_preserving_calibration import calibrate_dykstra

# Create example probability matrix (100 samples, 3 classes)
np.random.seed(42)
P = np.random.dirichlet([2, 1, 1], size=100)

# Define target column sums (must sum to number of samples)
M = np.array([40.0, 35.0, 25.0])  # sums to 100

# Calibrate probabilities
result = calibrate_dykstra(P, M)

# Access calibrated probabilities
calibrated_probs = result.Q

print(f"Original column sums: {P.sum(axis=0)}")
print(f"Target column sums: {M}")
print(f"Calibrated column sums: {calibrated_probs.sum(axis=0)}")
print(f"Converged in {result.iterations} iterations")
```

## Understanding the Result

The {class}`~rank_preserving_calibration.CalibrationResult` object contains:

* `Q`: The calibrated probability matrix
* `iterations`: Number of iterations until convergence
* `converged`: Whether the algorithm converged
* `final_change`: Final relative change between iterations

```python
# Check convergence
if result.converged:
    print("Algorithm converged successfully")
else:
    print(f"Algorithm did not converge (change: {result.final_change:.2e})")
```

## Alternative Algorithm: ADMM

You can also use the ADMM algorithm, which provides convergence history:

```python
from rank_preserving_calibration import calibrate_admm

result = calibrate_admm(P, M, verbose=True)

# ADMM result includes convergence history
print(f"Primal residuals: {result.primal_residuals[-5:]}")  # Last 5 values
print(f"Dual residuals: {result.dual_residuals[-5:]}")
```

## Nearly Isotonic Calibration

For more flexible calibration that allows small violations of isotonic constraints:

```python
# Epsilon-slack approach (recommended)
nearly_params = {"mode": "epsilon", "eps": 0.05}
result = calibrate_dykstra(P, M, nearly=nearly_params)

# Lambda-penalty approach (experimental)
nearly_params = {"mode": "lambda", "lam": 1.0}
result = calibrate_admm(P, M, nearly=nearly_params)
```

## Working with Real Data

Here's a more realistic example with classifier outputs:

```python
# Simulated classifier probabilities (overconfident)
n_samples, n_classes = 1000, 4

# Create overconfident predictions
logits = np.random.normal(0, 2, (n_samples, n_classes))
logits[:, 0] += 1  # bias toward class 0

# Convert to probabilities
P = np.exp(logits)
P = P / P.sum(axis=1, keepdims=True)

# True class proportions (from validation set)
true_proportions = np.array([0.3, 0.25, 0.25, 0.2])
M = true_proportions * n_samples

# Calibrate
result = calibrate_dykstra(P, M, max_iters=5000, tol=1e-8)

print(f"Original accuracy: {np.mean(P.argmax(axis=1) == np.arange(n_samples) % n_classes):.3f}")
print(f"Rank correlation preserved: {np.corrcoef(P.max(axis=1), result.Q.max(axis=1))[0,1]:.3f}")
```

## Common Parameters

Key parameters for both algorithms:

* `max_iters`: Maximum iterations (3000 for Dykstra, 1000 for ADMM)
* `tol`: Convergence tolerance (1e-7 for Dykstra, 1e-6 for ADMM)
* `verbose`: Print progress information
* `rtol`: Relative tolerance for isotonic regression (1e-12)

```python
result = calibrate_dykstra(
    P, M,
    max_iters=5000,
    tol=1e-8,
    verbose=True,
    rtol=1e-10
)
```

## Next Steps

* Learn about the {doc}`theory` behind rank-preserving calibration
* Explore detailed {doc}`examples` and use cases
* Check the full {doc}`api` reference