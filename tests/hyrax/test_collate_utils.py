import numpy as np

from hyrax.datasets.collate_utils import collate_as_1d_light_curve


def test_collate_as_1d_light_curve():
    """Test that the utility function to collate raw one-dimensional light-curves"""
    samples = [{"A": [0, 1, 2]}, {"A": [0, 1, 2]}, {"A": [0, 1, 2]}]

    expected_after_collate = {
        "A": np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]),
        "A_lengths": np.array([3, 3, 3]),
        "A_mask": np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
    }

    results = collate_as_1d_light_curve(samples, "A")
    np.testing.assert_equal(results, expected_after_collate)

    samples = [{"A": [0, 1, 2]}, {"A": [0, 1]}, {"A": [0]}]

    expected_after_collate = {
        "A": np.array([[0, 1, 2], [0, 1, 0], [0, 0, 0]]),
        "A_lengths": np.array([3, 2, 1]),
        "A_mask": np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]]),
    }

    results = collate_as_1d_light_curve(samples, "A")
    np.testing.assert_equal(results, expected_after_collate)
