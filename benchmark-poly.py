#!/usr/bin/env python


import numpy as np
import timeit
import os

from tail.timestream.poly_filter_array import poly_filter_array
from tests.timestream.test_poly_filter_array import simulate_poly_input

if __name__ == "__main__":
    num_threads = int(os.environ['OMP_NUM_THREADS'])
    number = 10
    repeat = 3

    number = 1
    repeat = 1

    # timing ground_template_filter_array
    nCh = 50000
    nTime = 20000
    nPoly = 4

    poly_input = simulate_poly_input(nCh, nTime, nPoly)
    temp = timeit.repeat(stmt='poly_filter_array(*poly_input, num_threads=num_threads)', repeat=repeat, number=number, globals=globals())
    print('{},{}'.format(num_threads, min(temp) / number))
