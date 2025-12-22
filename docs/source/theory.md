# Mathematical Theory

This section explains the mathematical foundations of rank-preserving calibration.

## Problem Formulation

Given a probability matrix $P \in \mathbb{R}^{N \times J}$ where each row represents a probability distribution over $J$ classes for $N$ samples, we want to find a calibrated matrix $Q$ that satisfies:

1. **Row constraints**: Each row is a valid probability distribution

   $$\sum_{j=1}^J Q_{ij} = 1, \quad Q_{ij} \geq 0 \quad \forall i,j$$

2. **Column constraints**: Column sums match target marginals

   $$\sum_{i=1}^N Q_{ij} = M_j \quad \forall j$$

3. **Isotonic constraints**: Within each column, values are non-decreasing when sorted by original scores

   $$Q_{i_1,j} \leq Q_{i_2,j} \text{ if } P_{i_1,j} \leq P_{i_2,j}$$

The optimization problem is:

$$\min_Q \|Q - P\|_F^2 \text{ subject to row, column, and isotonic constraints}$$

## Algorithmic Approaches

### Dykstra's Alternating Projections

Dykstra's method alternates between projecting onto different constraint sets while maintaining memory terms to ensure convergence to the intersection.

**Algorithm:**

1. Initialize $Q^{(0)} = P$, $U^{(0)} = 0$, $V^{(0)} = 0$

2. For $k = 0, 1, 2, \ldots$:

   a. **Row projection**: Project each row onto the probability simplex

      $$\tilde{Q}^{(k+1)} = \text{proj}_{\text{simplex}}(Q^{(k)} - U^{(k)})$$

      Update memory: $U^{(k+1)} = \tilde{Q}^{(k+1)} - (Q^{(k)} - U^{(k)})$

   b. **Column projection**: Project each column to satisfy isotonic and sum constraints

      $$Q^{(k+1)} = \text{proj}_{\text{isotonic}}(\tilde{Q}^{(k+1)} - V^{(k)})$$

      Update memory: $V^{(k+1)} = Q^{(k+1)} - (\tilde{Q}^{(k+1)} - V^{(k)})$

3. Check convergence: $\|Q^{(k+1)} - Q^{(k)}\|_F < \text{tol}$

**Simplex Projection:**

For a vector $x \in \mathbb{R}^J$, the projection onto the probability simplex is:

$$\text{proj}_{\text{simplex}}(x) = \max(0, x - \lambda \mathbf{1})$$

where $\lambda$ is chosen so that $\sum_j \max(0, x_j - \lambda) = 1$.

**Isotonic Projection:**

Uses the Pool Adjacent Violators (PAV) algorithm to find the isotonic regression that minimizes squared distance while satisfying the sum constraint.

### ADMM Optimization

The Alternating Direction Method of Multipliers (ADMM) formulates the problem with auxiliary variables and Lagrange multipliers.

**Augmented Lagrangian:**

$$L_\rho(Q, Z, \Lambda) = \frac{1}{2}\|Q - P\|_F^2 + \Lambda^T(AQ - b) + \frac{\rho}{2}\|AQ - b\|_2^2$$

where $A$ encodes the linear constraints and $b$ the target values.

**Algorithm:**

1. $Q$-update: Solve the quadratic subproblem
2. $Z$-update: Apply isotonic projections
3. $\Lambda$-update: Dual variable update

## Nearly Isotonic Calibration

Strict isotonic constraints can be overly restrictive. We provide two relaxation approaches:

### Epsilon-Slack Approach

Allow violations up to $\epsilon$:

$$Q_{i_1,j} \leq Q_{i_2,j} + \epsilon \text{ if } P_{i_1,j} \leq P_{i_2,j}$$

This maintains convexity and uses Euclidean projection onto the relaxed constraint set.

### Lambda-Penalty Approach

Add a penalty term for violations:

$$\min_Q \frac{1}{2}\|Q - P\|_F^2 + \lambda \sum_{j,i_1,i_2} \max(0, Q_{i_1,j} - Q_{i_2,j})$$

where the sum is over pairs with $P_{i_1,j} \leq P_{i_2,j}$.

## Convergence Properties

**Dykstra's Method:**
- Guaranteed convergence to the intersection of constraint sets
- Linear convergence rate under regularity conditions
- Each iteration maintains feasibility of the most recent constraint

**ADMM:**
- Convergence under standard assumptions (constraint qualification)
- Convergence rate depends on penalty parameter $\rho$
- Provides primal and dual residual tracking

**Nearly Isotonic:**
- Epsilon-slack: Maintains theoretical guarantees of Dykstra's method
- Lambda-penalty: Convergence depends on penalty parameter choice

## Computational Complexity

**Per Iteration:**
- Row projections: $O(NJ \log J)$ using efficient simplex projection
- Column projections: $O(NJ)$ using PAV algorithm
- Overall: $O(NJ \log J)$ per iteration

**Memory Requirements:**
- $O(NJ)$ for probability matrices
- Dykstra's method: Additional $O(NJ)$ for memory terms
- ADMM: Additional storage for auxiliary variables and dual variables

## Numerical Considerations

**Stability:**
- All computations use double precision (float64)
- Isotonic regression includes numerical safeguards
- Input validation checks for NaN/infinite values

**Feasibility:**
- The intersection of constraints may be empty if $\sum_j M_j \neq N$
- Warnings are issued when $|\sum_j M_j - N|$ is large
- Nearly isotonic approaches can help with infeasible problems
