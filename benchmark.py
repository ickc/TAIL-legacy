#!/usr/bin/env python


import numpy as np
import timeit

import csv
import argparse

from tail.timestream.ground_template_filter_array import ground_template_filter_array
from tests.timestream.test_ground_template_filter_array import simulate_ground_input

from tail.timestream.boundary_distance import boundary_distance
from tests.timestream.test_boundary_distance import circular_mask

if __name__ == "__main__":
    # local
    # process_range = (1, 2, 4, 8)
    # number = 1
    # repeat = 1

    # KNL
    # testing KNL's 1 to 68 cores, and 1 to 4 hyperthreadings
    process_range = (1, 2, 4, 8, 16, 32, 64, 65, 66, 67, 68,
                     128, 130, 132, 134, 136,
                     192, 195, 198, 201, 204,
                     256, 260, 264, 268, 272)
    number = 10
    repeat = 3

    # initialize time for output
    p_combinations = len(process_range)
    time = np.zeros([p_combinations, 3])
    time[:, 0] = process_range

    # timing ground_template_filter_array
    nCh = 50000
    nTime = 10000
    nPix = 8192

    ground_input = simulate_ground_input(nCh, nTime, nPix)
    for i, p in enumerate(process_range):
        temp = timeit.repeat(stmt='ground_template_filter_array(*ground_input, num_threads=p)', repeat=repeat, number=number, globals=globals())
        time[i, 1] = min(temp) / number
        print(time)

    # timing boundary_distance
    nPix = 1024

    boundary_input = (circular_mask(nPix, nPix // 2 - 2))
    for i, p in enumerate(process_range):
        temp = timeit.repeat(stmt='boundary_distance(boundary_input, num_threads=p)', repeat=repeat, number=number, globals=globals())
        time[i, 2] = min(temp) / number
        print(time)

    np.savetxt('benchmark.csv', time, delimiter=',', newline='\n',
               header='No. of threads,Ground Template (s), Boundary Distance (s)')
