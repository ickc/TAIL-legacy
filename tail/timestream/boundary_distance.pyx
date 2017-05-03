import numpy as np
cimport numpy as np

cimport cython

from libc.stdlib cimport malloc, calloc, free

from libcpp cimport bool

class Turtle(object):
    def __init__(self, i, j):
        self.x = i
        self.y = j
        self.v_x = 0
        self.v_y = 1
    def walk(self):
        self.x += self.v_x
        self.y += self.v_y
    def turn_left(self):
        # (x + yi) * -i = y - xi
        v_x = self.v_x
        self.v_x = self.v_y
        self.v_y = -v_x
    def turn_right(self):
        # (x + yi) * i = -y + xi
        v_x = self.v_x
        self.v_x = -self.v_y
        self.v_y = v_x
    def get_x(self):
        return self.x
    def get_y(self):
        return self.y

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef get_start(np.ndarray[np.uint8_t, cast=True, ndim=2] mask):
    m = mask.shape[0]
    n = mask.shape[1]
    for i in range(m):
        for j in range(n):
            if mask[i, j]:
                return complex(i, j)



@cython.boundscheck(False)
@cython.wraparound(False)
def get_boundary(mask, debug=False, stop=10):
    start = get_start(mask)
    boundary_list = [start]
    boundary_set = {start}
    turtle = Turtle(start.real, start.imag)
    if debug:
        counter = 0
    while True:
        if debug:
            print(turtle.get_loc())
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
            loc = complex(x, y)
            if loc not in boundary_set:
                boundary_list.append(loc)
                boundary_set.add(loc)
        if debug:
            counter += 1
            if counter == stop:
                break
        if loc == start:
            break
    return boundary_list