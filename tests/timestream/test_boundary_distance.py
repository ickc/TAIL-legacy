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
    boundary = get_boundary(mask, debug=False)
    assert boundary == [(2+2j), (3+2j), (4+2j), (4+3j), (4+4j), (3+4j), (2+4j), (2+3j)]
