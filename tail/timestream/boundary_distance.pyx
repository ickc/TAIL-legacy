import numpy as np
cimport numpy as np

cimport cython

# from libc.stdlib cimport malloc, calloc, free

from libcpp.vector cimport vector

# from libcpp cimport bool

# from numpy.math cimport INFINITY

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline Py_ssize_t _get_start(np.uint8_t* mask, Py_ssize_t mn):
    cdef Py_ssize_t ij
    for ij in range(mn):
        if mask[ij]:
            return ij

cdef class Coordinate(object):
    cdef Py_ssize_t x, y
    def __cinit__(Coordinate self, Py_ssize_t i, Py_ssize_t j):
        self.x = i
        self.y = j
    cdef void iadd(Coordinate self, Coordinate r2):
        self.x += r2.x
        self.y += r2.y
    # cdef bool isEqual(Coordinate self, Coordinate r2):
    #     return (self.x == r2.x and self.y == r2.y)
    cdef void rotate_left(Coordinate self):
        # (x + yi) * -i = y - xi
        cdef Py_ssize_t x = self.x
        self.x = self.y
        self.y = -x
    cdef void rotate_right(Coordinate self):
        # (x + yi) * i = -y + xi
        cdef Py_ssize_t x = self.x
        self.x = -self.y
        self.y = x
    cdef Py_ssize_t get_x(Coordinate self):
        return self.x
    cdef Py_ssize_t get_y(Coordinate self):
        return self.y

cdef class Turtle(object):
    cdef Coordinate r, v
    def __cinit__(Turtle self, Coordinate c2):
        self.r = c2
        self.v = Coordinate(1, 0)
    cdef void walk(Turtle self):
        self.r.iadd(self.v)
    cdef void rotate_left(Turtle self):
        self.v.rotate_left()
    cdef void rotate_right(Turtle self):
        self.v.rotate_right()
    cdef Py_ssize_t get_x(Turtle self):
        return self.r.get_x()
    cdef Py_ssize_t get_y(Turtle self):
        return self.r.get_y()

# cdef class Boundary()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef vector[Py_ssize_t] _get_boundary(np.ndarray[np.uint8_t, cast=True, ndim=2] mask):
    cdef Py_ssize_t m, n, start, x, y, loc
    cdef vector[Py_ssize_t] boundary

    m = mask.shape[0]
    n = mask.shape[1]
    start = _get_start(&mask[0, 0], m * n)

    turtle = Turtle(Coordinate(start // m, start % m))
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

# @cython.boundscheck(False)
# @cython.wraparound(False)
# def get_box(vector[Py_ssize_t] boundary, Py_ssize_t m, Py_ssize_t n):
#     Py_ssize_t i_min =
#     for k in boundary:
        

def get_boundary(np.ndarray[np.uint8_t, cast=True, ndim=2] mask):
    cdef vector[Py_ssize_t] boundary = _get_boundary(mask)
    return np.asarray(boundary)

# def boundary_distance(np.ndarray[np.uint8_t, cast=True, ndim=2] mask):
    