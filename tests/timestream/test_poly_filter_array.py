"""
"""
from .context import poly_filter_array
import pickle
from tests.helper import assertIdenticalList
import sys
py2 = sys.version_info[0] == 2


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
    poly_filter_array(*poly_filter_array_input)
    assertIdenticalList(poly_filter_array_input, poly_filter_array_output)
