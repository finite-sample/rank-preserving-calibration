"""Helper utilities used in examples and tests.

This module contains functions for generating synthetic data sets and
analyzing calibration results.  They are kept outside of the main
package to avoid pulling example dependencies into the installation.
"""

import numpy as np

__all__ = [
    "analyze_calibration_result",
    "create_realistic_classifier_case",
    "create_survey_reweighting_case",
    "create_test_case",
]


def create_test_case(
    case_type: str, N: int = 50, J: int = 4, seed: int | None = None, **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic test cases for calibration algorithms.

    Parameters
    ----------
    case_type : str
        Type of test case to create:
        - 'random': Random probabilities from Dirichlet distribution
        - 'skewed': Some classes more likely than others
        - 'linear': Linear trends for testing rank preservation
        - 'challenging': Difficult case with potential feasibility issues
    N : int, default=50
        Number of instances (rows).
    J : int, default=4
        Number of classes (columns).
    seed : int, optional
        Random seed for reproducibility.
    **kwargs
        Additional parameters for specific case types:
        - concentration: Dirichlet concentration for 'random' (default=1.0)
        - skew_factor: Factor for skewed classes (default=3.0)
        - noise_level: Noise level for 'linear' trends (default=0.1)
        - infeasibility_level: Level of infeasibility for 'challenging' (default=0.2)

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (P, M) where P is the probability matrix and M is target marginals.

    Examples
    --------
    >>> P, M = create_test_case("random", N=100, J=4, seed=42)
    >>> P_skewed, M_skewed = create_test_case("skewed", N=50, J=3, skew_factor=2.0)
    """
    # Optional scipy import for test case generation
    try:
        from scipy.stats import dirichlet as sp_dirichlet
    except ImportError:  # pragma: no cover - SciPy may not be installed
        sp_dirichlet = None

    rng = np.random.default_rng(seed)

    def _dirichlet(alpha, size):
        if sp_dirichlet is not None:
            return sp_dirichlet.rvs(alpha, size=size, random_state=rng)
        return rng.dirichlet(alpha, size=size)

    if case_type == "random":
        concentration = kwargs.get("concentration", 1.0)
        alpha = np.full(J, concentration)
        P = _dirichlet(alpha, size=N)
        M = np.full(J, N / J)  # Equal marginals

    elif case_type == "skewed":
        skew_factor = kwargs.get("skew_factor", 3.0)
        alpha = np.ones(J)
        alpha[0] *= skew_factor  # Make first class more likely
        alpha[-1] *= skew_factor  # Make last class more likely
        P = _dirichlet(alpha, size=N)
        M = np.full(J, N / J)

    elif case_type == "linear":
        noise_level = kwargs.get("noise_level", 0.1)
        P = np.zeros((N, J))

        for j in range(J):
            # Create different linear trends for each class
            slope = 0.8 / N * (1 + 0.5 * j)
            intercept = 0.1 + 0.1 * j
            trend = slope * np.arange(N) + intercept

            # Add noise
            if noise_level > 0:
                trend += noise_level * rng.standard_normal(N)

            trend = np.clip(trend, 0.01, 0.99)
            P[:, j] = trend

        # Normalize rows to sum to 1
        P = P / P.sum(axis=1, keepdims=True)
        M = np.full(J, N / J)

    elif case_type == "challenging":
        infeasibility_level = kwargs.get("infeasibility_level", 0.2)
        # Create extreme probabilities using beta distribution
        P = rng.beta(0.3, 0.3, size=(N, J))
        P = P / P.sum(axis=1, keepdims=True)

        # Create target marginals with controlled infeasibility
        base_marginals = N / J * np.ones(J)
        bias = rng.uniform(-0.5, 0.5, J)
        bias = bias - bias.mean()  # Center the bias
        M = base_marginals + infeasibility_level * N * bias
        M = np.maximum(M, 0.1)  # Ensure positive marginals

    else:
        raise ValueError(
            f"Unknown case type: {case_type}. "
            "Choose from: 'random', 'skewed', 'linear', 'challenging'"
        )

    return P, M


def create_realistic_classifier_case(
    N: int = 500,
    J: int = 4,
    miscalibration_type: str = "overconfident",
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create realistic classifier probabilities that need calibration.

    Simulates a multiclass classifier (e.g., image classification, NLP) where
    the predicted probabilities are systematically biased.

    Parameters
    ----------
    N : int
        Number of samples.
    J : int
        Number of classes.
    miscalibration_type : str
        Type of miscalibration: 'overconfident', 'underconfident', 'biased'.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict]
        P (predicted probabilities), M (true class frequencies), info dict.
    """
    # Optional scipy import
    try:
        from scipy.stats import dirichlet as sp_dirichlet
    except ImportError:  # pragma: no cover - SciPy may not be installed
        sp_dirichlet = None

    rng = np.random.default_rng(seed)

    def _dirichlet(alpha, size):
        if sp_dirichlet is not None:
            return sp_dirichlet.rvs(alpha, size=size, random_state=rng)
        return rng.dirichlet(alpha, size=size)

    # Simulate true class labels with realistic imbalance
    if J <= 4:
        base_probs = np.array([0.4, 0.3, 0.2, 0.1])[:J]
    else:
        # Generate decreasing probabilities for more classes
        base_probs = np.array([0.4 * (0.7**i) for i in range(J)])

    true_class_probs = base_probs / base_probs.sum()

    true_labels = rng.choice(J, size=N, p=true_class_probs)
    M = np.bincount(true_labels, minlength=J).astype(float)

    # Generate classifier predictions with systematic bias
    P = np.zeros((N, J))

    for i in range(N):
        true_class = true_labels[i]

        if miscalibration_type == "overconfident":
            correct_prob = rng.beta(8, 2)  # High confidence when correct
            remaining = 1 - correct_prob
            other_probs = _dirichlet([0.5] * (J - 1), size=1)[0] * remaining

        elif miscalibration_type == "underconfident":
            correct_prob = rng.beta(2, 2) * 0.6 + 0.3  # 0.3 to 0.9 range
            remaining = 1 - correct_prob
            other_probs = _dirichlet([1.5] * (J - 1), size=1)[0] * remaining

        elif miscalibration_type == "biased":
            if true_class == 0:  # Underdetects class 0
                correct_prob = rng.beta(2, 3)
            elif true_class == 1:  # Overdetects class 1
                correct_prob = rng.beta(5, 1)
            else:
                correct_prob = rng.beta(3, 2)

            remaining = 1 - correct_prob
            other_probs = _dirichlet([1] * (J - 1), size=1)[0] * remaining

        # Build probability vector
        prob_vector = np.zeros(J)
        prob_vector[true_class] = correct_prob
        other_indices = [j for j in range(J) if j != true_class]
        prob_vector[other_indices] = other_probs

        P[i] = prob_vector

    info = {
        "scenario": "classifier_calibration",
        "miscalibration_type": miscalibration_type,
        "true_class_distribution": true_class_probs,
        "observed_accuracy": np.mean(np.argmax(P, axis=1) == true_labels),
        "mean_confidence": np.mean(np.max(P, axis=1)),
        "true_labels": true_labels,
    }

    return P, M, info


def create_survey_reweighting_case(
    N: int = 1000, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create survey data that needs demographic reweighting.

    Simulates a political poll or market research survey where the sample
    doesn't match the target population demographics.

    Parameters
    ----------
    N : int
        Number of survey respondents.
    seed : int, optional
        Random seed.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict]
        P (survey responses), M (target demographics), info dict.
    """
    # Optional scipy import
    try:
        from scipy.stats import dirichlet as sp_dirichlet
    except ImportError:  # pragma: no cover - SciPy may not be installed
        sp_dirichlet = None

    rng = np.random.default_rng(seed)

    def _dirichlet(alpha, size):
        if sp_dirichlet is not None:
            return sp_dirichlet.rvs(alpha, size=size, random_state=rng)
        return rng.dirichlet(alpha, size=size)

    # Define demographic categories: [Urban, Suburban, Rural, Other]
    true_demographics = np.array([0.35, 0.45, 0.18, 0.02])
    sample_demographics = np.array([0.55, 0.30, 0.12, 0.03])  # Biased toward urban

    respondent_types = rng.choice(4, size=N, p=sample_demographics)

    # Response patterns differ by demographic group
    response_patterns = {
        0: [0.42, 0.38, 0.15, 0.05],  # Urban: leans A
        1: [0.35, 0.45, 0.15, 0.05],  # Suburban: leans B
        2: [0.25, 0.50, 0.20, 0.05],  # Rural: strongly leans B
        3: [0.30, 0.30, 0.25, 0.15],  # Other: more undecided
    }

    P = np.zeros((N, 4))
    for i in range(N):
        demo_type = respondent_types[i]
        base_probs = np.array(response_patterns[demo_type])
        # Add individual variation
        noise = _dirichlet([10] * 4, size=1)[0] * 0.3
        individual_probs = 0.7 * base_probs + 0.3 * noise
        P[i] = individual_probs / individual_probs.sum()

    # Target marginals: what the poll should show if it matched true demographics
    M = np.zeros(4)
    for demo in range(4):
        demo_count = true_demographics[demo] * N
        M += demo_count * np.array(response_patterns[demo])

    info = {
        "scenario": "survey_reweighting",
        "true_demographics": true_demographics,
        "sample_demographics": sample_demographics,
        "sample_bias": sample_demographics - true_demographics,
        "response_patterns": response_patterns,
        "raw_results": P.mean(axis=0),
        "target_results": M / N,
    }

    return P, M, info


def analyze_calibration_result(P: np.ndarray, result, M: np.ndarray) -> dict:
    """Analyze the impact of calibration on probability matrix.

    Parameters
    ----------
    P : np.ndarray
        Original probability matrix.
    result : CalibrationResult or ADMMResult
        Calibration result object.
    M : np.ndarray
        Target marginals.

    Returns
    -------
    dict
        Analysis metrics including prediction changes, confidence shifts, etc.
    """
    Q = result.Q

    # Basic constraint satisfaction
    row_sums = Q.sum(axis=1)
    col_sums = Q.sum(axis=0)

    # Prediction changes
    original_preds = np.argmax(P, axis=1)
    calibrated_preds = np.argmax(Q, axis=1)
    prediction_changes = np.mean(original_preds != calibrated_preds)

    # Confidence changes
    original_confidence = np.mean(np.max(P, axis=1))
    calibrated_confidence = np.mean(np.max(Q, axis=1))

    # Distribution changes
    original_entropy = np.mean(-np.sum(P * np.log(P + 1e-10), axis=1))
    calibrated_entropy = np.mean(-np.sum(Q * np.log(Q + 1e-10), axis=1))

    # Marginal correction magnitude
    marginal_correction = np.abs(P.sum(axis=0) - M).sum()

    return {
        "constraint_satisfaction": {
            "max_row_error": np.max(np.abs(row_sums - 1.0)),
            "max_col_error": np.max(np.abs(col_sums - M)),
            "rank_preservation": result.max_rank_violation,
        },
        "prediction_impact": {
            "prediction_changes": prediction_changes,
            "original_confidence": original_confidence,
            "calibrated_confidence": calibrated_confidence,
            "confidence_change": calibrated_confidence - original_confidence,
        },
        "distribution_impact": {
            "original_entropy": original_entropy,
            "calibrated_entropy": calibrated_entropy,
            "entropy_change": calibrated_entropy - original_entropy,
            "total_change": np.linalg.norm(Q - P),
            "relative_change": np.linalg.norm(Q - P) / np.linalg.norm(P),
        },
        "marginal_correction": marginal_correction,
        "convergence": {
            "converged": True,  # Always True if we get a result (failure raises CalibrationError)
            "iterations": result.iterations,
            "final_change": result.final_change,
        },
    }
