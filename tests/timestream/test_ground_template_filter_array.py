"""
"""
from .context import ground_template_filter_array
import pickle
from tests.helper import assertIdenticalList
import numpy as np
import sys
py2 = sys.version_info[0] == 2


def simulate_ground_input(nCh, nTime, nPix):
    az_time_width = 400
    az_min = 3.5
    az_max = 4.
    az_range = az_max - az_min

    input_array = (np.random.rand(nCh, nTime) - 0.5) * 0.2

    az = np.array([(az_range * (1 -
                                abs(i % (2 * az_time_width) / float(az_time_width) - 1)) + az_min
                    ) for i in range(nTime)])

    mask_half_width = 50
    mask = np.repeat([[(i % az_time_width < mask_half_width
                        or i % az_time_width > az_time_width - mask_half_width
                        ) for i in range(nTime)]], nCh, axis=0)

    pixel_size = az_range / nPix

    return input_array, az, mask, pixel_size


def test_ground_template_filter_array():
    with open('tests/timestream/ground_template_filter_array_input.pkl', 'rb') as f:
        if not py2:
            ground_template_filter_array_input = pickle.load(
                f, encoding='latin1')
        else:
            ground_template_filter_array_input = pickle.load(f)
    with open('tests/timestream/ground_template_filter_array_output.pkl', 'rb') as f:
        if not py2:
            ground_template_filter_array_output = pickle.load(
                f, encoding='latin1')
        else:
            ground_template_filter_array_output = pickle.load(f)
    ground_template_filter_array(*ground_template_filter_array_input)
    assertIdenticalList(
        ground_template_filter_array_input,
        ground_template_filter_array_output)
