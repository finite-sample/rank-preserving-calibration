# tests/test_ovr_isotonic.py
import numpy as np

from rank_preserving_calibration import calibrate_ovr_isotonic


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
