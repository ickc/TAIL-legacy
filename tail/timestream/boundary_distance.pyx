import numpy as np
cimport numpy as np

cimport cython

from libc.stdlib cimport malloc, calloc, free

from libcpp.vector cimport vector

from libcpp cimport bool

cdef class Turtle(object):
    cdef Py_ssize_t x, y, v_x, v_y
    def __cinit__(self, Py_ssize_t i, Py_ssize_t j):
        self.x = i
        self.y = j
        self.v_x = 1
        self.v_y = 0
    cdef void walk(self):
        self.x += self.v_x
        self.y += self.v_y
    cdef void turn_left(self):
        # (x + yi) * -i = y - xi
        cdef Py_ssize_t v_x = self.v_x
        self.v_x = self.v_y
        self.v_y = -v_x
    cdef void turn_right(self):
        # (x + yi) * i = -y + xi
        cdef Py_ssize_t v_x = self.v_x
        self.v_x = -self.v_y
        self.v_y = v_x
    cdef Py_ssize_t get_x(self):
        return self.x
    cdef Py_ssize_t get_y(self):
        return self.y

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Py_ssize_t _get_start(np.uint8_t* mask, Py_ssize_t mn):
    cdef Py_ssize_t ij
    for ij in range(mn):
        if mask[ij]:
            return ij

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef vector[Py_ssize_t] _get_boundary(np.ndarray[np.uint8_t, cast=True, ndim=2] mask):
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
            turtle.turn_left()
        else:
            turtle.turn_right()
    return boundary

def get_boundary(np.ndarray[np.uint8_t, cast=True, ndim=2] mask):
    cdef vector[Py_ssize_t] boundary = _get_boundary(mask)
    return np.asarray(boundary)
