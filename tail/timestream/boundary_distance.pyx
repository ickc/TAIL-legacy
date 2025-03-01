# distutils: language = c++

import numpy as np
cimport numpy as np

cimport cython

from cython.parallel cimport prange

from libc.stdlib cimport malloc, calloc, free

from libc.math cimport sqrt

from libcpp.vector cimport vector

from libcpp cimport bool

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Py_ssize_t _get_start(np.uint8_t* mask, Py_ssize_t mn):
    cdef Py_ssize_t ij
    for ij in range(mn):
        if mask[ij]:
            return ij


cdef class Turtle(object):
    cdef Py_ssize_t x, y, v_x, v_y
    def __cinit__(Turtle self, Py_ssize_t x, Py_ssize_t y):
        self.x = x
        self.y = y
        self.v_x = 0
        self.v_y = 1
    cdef void walk(Turtle self):
        self.x += self.v_x
        self.y += self.v_y
    cdef void rotate_left(Turtle self):
        # (x + yi) * i = -y + xi
        cdef Py_ssize_t v_x = self.v_x
        self.v_x = -self.v_y
        self.v_y = v_x
    cdef void rotate_right(Turtle self):
        # (x + yi) * -i = y - xi
        cdef Py_ssize_t v_x = self.v_x
        self.v_x = self.v_y
        self.v_y = -v_x
    cdef Py_ssize_t get_x(Turtle self):
        return self.x
    cdef Py_ssize_t get_y(Turtle self):
        return self.y

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef vector[Py_ssize_t] get_boundary(np.ndarray[np.uint8_t, cast=True, ndim=2] mask):
    cdef Py_ssize_t m, n, start, x, y, loc
    cdef vector[Py_ssize_t] boundary

    m = mask.shape[0]
    n = mask.shape[1]
    start = _get_start(&mask[0, 0], m * n)

    turtle = Turtle(start // m, start % m)
    boundary.push_back(start)
    while True:
        turtle.walk()
        x = turtle.get_x()
        y = turtle.get_y()
        if mask[x, y]:
            loc = x * m + y
            if loc == start:
                break
            if loc != boundary.back():
                boundary.push_back(loc)
            turtle.rotate_left()
        else:
            turtle.rotate_right()
    return boundary

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def boundary_distance(np.ndarray[np.uint8_t, cast=True, ndim=2] mask, int num_threads=4, bool debug=False):

    cdef Py_ssize_t i, j, k, x, y, x_min, x_max, y_min, y_max

    cdef Py_ssize_t m = mask.shape[0]
    cdef Py_ssize_t n = mask.shape[1]

    x_min = m
    y_min = n
    x_max = 0
    y_max = 0

    cdef vector[Py_ssize_t] boundary = get_boundary(mask)
    cdef Py_ssize_t nBoundary = boundary.size()

    cdef Py_ssize_t* boundary_coordinate = <Py_ssize_t*>malloc(2 * nBoundary * sizeof(Py_ssize_t))

    # convert boundary from 1D indexing to 2D indexing
    # and obtain the smallest box that includes the boundary
    # Not vectorized, use the following if in C
    #pragma ivdep
    for k in range(nBoundary):
        x = boundary[k] // m
        y = boundary[k] % m
        boundary_coordinate[2 * k] = x
        boundary_coordinate[2 * k + 1] = y
        x_min = x if x < x_min else x_min
        y_min = y if y < y_min else y_min
        x_max = x if x > x_max else x_max
        y_max = y if y > y_max else y_max

    cdef Py_ssize_t* distances_sq = <Py_ssize_t*>calloc(m * n, sizeof(Py_ssize_t))
    # initialize
    cdef Py_ssize_t upper_bound = m * m + n * n
    for i in range(x_min + 1, x_max):
        # SIMD checked
        for j in range(y_min + 1, y_max):
            distances_sq[i * m + j] = upper_bound if mask[i, j] else 0

    cdef Py_ssize_t loc, distance_sq
    for i in prange(x_min + 1, x_max, nogil=True, schedule='guided', num_threads=num_threads):
        for j in range(y_min + 1, y_max):
            if mask[i, j]:
                loc = i * m + j
                # SIMD checked
                for k in range(nBoundary):
                    distance_sq = (i - boundary_coordinate[2 * k])**2 + (j - boundary_coordinate[2 * k + 1])**2
                    distances_sq[loc] = distance_sq if distance_sq < distances_sq[loc] else distances_sq[loc]

    cdef double[:, :] distances = np.zeros([m, n])
    cdef double* distances_ptr = &distances[0, 0]

    # SIMD checked
    for i in prange(x_min + 1, x_max, nogil=True, schedule='guided', num_threads=num_threads):
        for j in range(y_min + 1, y_max):
            loc = i * m + j
            distances_ptr[loc] = sqrt(<double>distances_sq[loc])

    # cdef np.int64_t[:, :] boundary_coordinate_np
    if debug:
        boundary_coordinate_np = np.empty([nBoundary * 2], dtype=np.int64)
        for i in range(nBoundary * 2):
            boundary_coordinate_np[i] = boundary_coordinate[i]


    free(boundary_coordinate)
    free(distances_sq)

    if debug:
        return (distances, np.reshape(boundary_coordinate_np, [nBoundary, 2]), np.array([x_min, y_min, x_max, y_max]))
    else:
        return np.asarray(distances)
