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

## Nearly Isotonic Calibration

Strict isotonic constraints can be overly restrictive. We provide two relaxation approaches:

### Epsilon-Slack Approach

Allow violations up to $\epsilon$:

$$Q_{i_1,j} \leq Q_{i_2,j} + \epsilon \text{ if } P_{i_1,j} \leq P_{i_2,j}$$

This maintains convexity and uses Euclidean projection onto the relaxed constraint set.

### Penalised Violations

`calibrate_soft` penalises violations instead of bounding them, and softens the marginals too:

$$\min_Q \frac{1}{2}\|Q - P\|_F^2 + \frac{\lambda_m}{2}\|Q^T\mathbf{1} - M\|^2 + \lambda_r \sum_j \sum_i \max(0, Q_{(i),j} - Q_{(i+1),j})$$

with each row of $Q$ on the simplex, where $Q_{(i),j}$ is column $j$ in the order of $P$'s column $j$. It is a quadratic program, with one hinge variable per adjacent pair standing in for each $\max(0, \cdot)$, solved by an interior-point method as `calibrate_qp` is. `calibrate_kl_soft` is the same idea with $\mathrm{KL}(Q\|R)$ in place of the squared error and hard marginals, solved exactly as an exponential-cone program.

## Convergence Properties

**Dykstra's Method:**
- Converges to the projection onto the intersection when that intersection is non-empty. Verified against an independent convex solver: the returned matrix matches a QP formulation of the whole problem to better than $10^{-9}$ in objective.
- Each iteration maintains feasibility of the most recent constraint, so no finite iterate satisfies *both* sets exactly — every sweep ends on a column projection, which perturbs the row sums. Convergence tolerances are therefore absolute and achievable (default $10^{-8}$) rather than aspirational.
- **The iteration count grows superlinearly in $N$.** Measured at $J = 4$ with feasible targets: 159 iterations at $N = 25$, 533 at $N = 50$, 6,982 at $N = 100$, and past 60,000 at $N = 200$. For context, a general-purpose QP solver handles the $N = 100$ case exactly in 0.06s against Dykstra's 5.4s. Treat $N$ in the low hundreds as the practical ceiling for the exact projection, and use the $\varepsilon$-relaxation above that.

**Nearly Isotonic:**
- Epsilon-slack: Maintains theoretical guarantees of Dykstra's method
- Penalised violations (`calibrate_soft`): one interior-point solve of a quadratic program, checked against an independent convex solver

**KL calibration:**
- `calibrate_kl_soft` is an exponential-cone program solved by an interior-point method. `calibrate_kl` solves the same program at a penalty weight large enough that the minimiser has no rank violations; because the penalised problem relaxes the constrained one, a violation-free penalised minimiser is the constrained minimiser. Posing the isotonic constraints directly stalls the solver when many are active at once.

## Computational Complexity

**Per Iteration:**
- Row projections: $O(NJ \log J)$ using efficient simplex projection
- Column projections: $O(NJ)$ using PAV algorithm
- Overall: $O(NJ \log J)$ per iteration

**Memory Requirements:**
- $O(NJ)$ for probability matrices
- Dykstra's method: Additional $O(NJ)$ for memory terms

## Numerical Considerations

**Stability:**
- All computations use double precision (float64)
- Isotonic regression includes numerical safeguards
- Input validation checks for NaN/infinite values

**Feasibility:**
- The intersection is empty **whenever** $\sum_j M_j \neq N$, not merely "may be". Row sums of one fix the grand total at $N$, so no matrix can satisfy both constraint sets, and there is no closest point satisfying both to fall back on. The solver warns and then raises.
- Conversely, when $\sum_j M_j = N$ the intersection is **never** empty: the constant matrix $Q_{ij} = M_j / N$ has unit row sums, column sums $M_j$, and weakly isotonic columns. Any failure in that regime is conditioning or iteration budget, never infeasibility.
- Warnings fire as soon as $|\sum_j M_j - N|$ exceeds floating-point slack. They previously fired only above $10\%$ of $N$, so a 2% mismatch produced no warning and then failed hard.
- Nearly isotonic approaches relax the *isotonic* constraint, which helps with ill-conditioning. They do not make $\sum_j M_j \neq N$ solvable — rescale the targets instead.
