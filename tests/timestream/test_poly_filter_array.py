"""
"""
from .context import poly_filter_array
import pickle
from tests.helper import assertIdenticalList


def test_poly_filter_array():
    with open('tests/timestream/poly_filter_array_input.pkl', 'rb') as f:
        poly_filter_array_input = pickle.load(f)
    with open('tests/timestream/poly_filter_array_output.pkl', 'rb') as f:
        poly_filter_array_output = pickle.load(f)
    poly_filter_array(*poly_filter_array_input)
    assertIdenticalList(poly_filter_array_input, poly_filter_array_output)
