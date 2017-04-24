"""
"""
from .context import ground_template_filter_array
import pickle
from tests.helper import assertIdenticalList


def test_ground_template_filter_array():
	with open('tests/timestream/ground_template_filter_array_input.pkl', 'rb') as f:
		ground_template_filter_array_input = pickle.load(f)
	with open('tests/timestream/ground_template_filter_array_output.pkl', 'rb') as f:
		ground_template_filter_array_output = pickle.load(f)
	ground_template_filter_array(*ground_template_filter_array_input)
	assertIdenticalList(ground_template_filter_array_input, ground_template_filter_array_output)
