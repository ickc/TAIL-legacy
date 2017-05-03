"""
"""
from .context import get_boundary
import numpy as np


def circular_mask(n, r):
    mask = np.zeros((n, n),dtype=bool)
    for i in range(n):
        for j in range(n):
            if (i - n / 2)**2 + (j - n / 2)**2 < r**2:
                mask[i][j]=True
    return mask

def test_get_boundary():
    mask = circular_mask(6, 2)
    boundary = get_boundary(mask)
    # "unpacked" version: [(2, 2), (3, 2), (4, 2), (4, 3), (4, 4), (3, 4), (2, 4), (2, 3)]
    np.testing.assert_array_equal(boundary, np.array([14, 20, 26, 27, 28, 22, 16, 15]))
