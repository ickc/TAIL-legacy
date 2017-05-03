import numpy as np
cimport numpy as np

cimport cython

from libc.stdlib cimport malloc, calloc, free

from libcpp cimport bool

cdef class Turtle(object):
    cdef Py_ssize_t x, y, v_x, v_y
    def __cinit__(self, Py_ssize_t i, Py_ssize_t j):
        self.x = i
        self.y = j
        self.v_x = 0
        self.v_y = 1
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void walk(self):
        self.x += self.v_x
        self.y += self.v_y
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void turn_left(self):
        # (x + yi) * -i = y - xi
        cdef Py_ssize_t v_x = self.v_x
        self.v_x = self.v_y
        self.v_y = -v_x
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void turn_right(self):
        # (x + yi) * i = -y + xi
        cdef Py_ssize_t v_x = self.v_x
        self.v_x = -self.v_y
        self.v_y = v_x
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef Py_ssize_t get_x(self):
        return self.x
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef Py_ssize_t get_y(self):
        return self.y



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def get_boundary(np.ndarray[np.uint8_t, cast=True, ndim=2] mask):
    cdef Py_ssize_t m, n, i, j, ij, start, x, y, loc

    m = mask.shape[0]
    n = mask.shape[1]
    for ij in range(m * n):
        i = ij // m
        j = ij % m
        if mask[i, j]:
            start = ij
            break

    boundary_list = [start]
    boundary_set = {start}
    turtle = Turtle(start // m, start % m)
    while True:
        x = turtle.get_x()
        y = turtle.get_y()
        if mask[x, y]:
            turtle.turn_left()
        else:
            turtle.turn_right()
        turtle.walk()
        x = turtle.get_x()
        y = turtle.get_y()
        if mask[x, y]:
            loc = x * m + y
            if loc not in boundary_set:
                boundary_list.append(loc)
                boundary_set.add(loc)
        if loc == start:
            break
    boundary_list = [(ij // m, ij % m) for ij in boundary_list]
    return boundary_list
