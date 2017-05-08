"""
"""
from .context import boundary_distance
import numpy as np


def circular_mask(n, r):
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if (i - n / 2) ** 2 + (j - n / 2) ** 2 < r ** 2:
                mask[i][j] = True
    return mask


def test_boundary_distance():
    mask = circular_mask(6, 2)
    apodization, boundary, box = boundary_distance(mask, debug=True)
    np.testing.assert_array_equal(boundary, np.array(
        [[2, 2], [2, 3], [2, 4], [3, 4], [4, 4], [4, 3], [4, 2], [3, 2]]))
    np.testing.assert_array_equal(box, np.array([2, 2, 4, 4]))
    np.testing.assert_array_equal(apodization, np.array(
        [[0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0.],
         [0., 0., 0., 0., 0., 0.],
         [0., 0., 0., 0., 0., 0.]]))
