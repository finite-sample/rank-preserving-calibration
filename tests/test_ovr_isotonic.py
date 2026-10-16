# tests/test_ovr_isotonic.py
import numpy as np
import pytest

from rank_preserving_calibration import CalibrationError, calibrate_ovr_isotonic


def test_ovr_isotonic_basic():
    """Test basic functionality of the One-vs-Rest isotonic calibration."""
    y = np.array([0, 1, 2, 0, 1, 2])
    probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.2, 0.2],
            [0.3, 0.6, 0.1],
            [0.2, 0.2, 0.6],
        ]
    )

    result = calibrate_ovr_isotonic(y, probs)
    calibrated_probs = result["Q"]

    # Check shape
    assert calibrated_probs.shape == probs.shape

    # Check that rows sum to 1
    np.testing.assert_allclose(calibrated_probs.sum(axis=1), 1.0, rtol=1e-6)

    # Check that values are non-negative
    assert np.all(calibrated_probs >= 0)


def test_ovr_isotonic_calibration_effect():
    """Test that the calibration has an effect on the probabilities."""
    y = np.array([0, 0, 1, 1])
    # Probabilities are anti-calibrated for class 0
    probs = np.array(
        [
            [0.8, 0.2],
            [0.7, 0.3],
            [0.3, 0.7],
            [0.2, 0.8],
        ]
    )

    result = calibrate_ovr_isotonic(y, probs)
    calibrated_probs = result["Q"]

    # The calibrated probabilities should be different from the original ones.
    assert not np.allclose(probs, calibrated_probs)

    # For class 0, the probabilities should be adjusted upwards for the first two
    # samples (which have label 0) and downwards for the last two.
    assert calibrated_probs[0, 0] > probs[0, 0]
    assert calibrated_probs[1, 0] > probs[1, 0]
    assert calibrated_probs[2, 0] < probs[2, 0]
    assert calibrated_probs[3, 0] < probs[3, 0]


def test_ovr_isotonic_matches_sklearn_with_tied_scores():
    """Per-class fits must equal scikit-learn's IsotonicRegression.

    Scores on a 0.01 grid tie heavily. Tied scores are pooled into weighted
    points, which is the case the weighted PAV used to get wrong (off by 0.31).
    """
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(0)
    probs = rng.multinomial(100, [0.5, 0.3, 0.2], size=200) / 100
    y = np.array([rng.choice(3, p=p) for p in probs])

    expected = np.column_stack(
        [
            IsotonicRegression().fit(probs[:, j], y == j).predict(probs[:, j])
            for j in range(3)
        ]
    )
    expected /= expected.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(
        calibrate_ovr_isotonic(y, probs)["Q"], expected, atol=1e-12
    )


@pytest.mark.parametrize(
    "y", [[0, -1], [0, 2], [0.0, 0.5], [0], [1j, 1 + 0j], [True, False]]
)
def test_ovr_isotonic_rejects_labels_outside_the_classes(y):
    # A label outside [0, J) left a row of zeros and returned NaN.
    with pytest.raises(CalibrationError, match="integer labels"):
        calibrate_ovr_isotonic(np.array(y), np.array([[0.9, 0.1], [0.1, 0.9]]))
