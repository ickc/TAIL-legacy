"""
"""
from .context import poly_filter_array
import pickle
from tests.helper import assertIdenticalList
import sys
py2 = sys.version_info[0] == 2
import numpy as np

def simulate_poly_input(nCh, nTime, polyorder):
    az_time_width = 400

    input_array = (np.random.rand(nCh, nTime) - 0.5) * 0.2

    mask_half_width = 50
    mask = np.repeat([[(i % az_time_width < mask_half_width
                        or i % az_time_width > az_time_width - mask_half_width
                        ) for i in range(nTime)]], nCh, axis=0)

    scan0 = np.arange(171, 10000, 408)
    scan1 = np.full_like(scan0, 397)
    scan_list = np.dstack((scan0, scan1))[0]

    return input_array, mask, mask, scan_list, 0, polyorder


def test_poly_filter_array():
    with open('tests/timestream/poly_filter_array_input.pkl', 'rb') as f:
        if not py2:
            poly_filter_array_input = pickle.load(f, encoding='latin1')
        else:
            poly_filter_array_input = pickle.load(f)
    with open('tests/timestream/poly_filter_array_output.pkl', 'rb') as f:
        if not py2:
            poly_filter_array_output = pickle.load(f, encoding='latin1')
        else:
            poly_filter_array_output = pickle.load(f)
    coeff_out = poly_filter_array(*poly_filter_array_input)
    assertIdenticalList(poly_filter_array_input, poly_filter_array_output)
    return coeff_out
