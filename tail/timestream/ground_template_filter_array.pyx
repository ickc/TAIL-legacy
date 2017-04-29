import numpy as np
cimport numpy as np
import cython

from cython.view cimport array as cvarray
from cpython cimport bool
from cython.parallel import parallel, prange

from libc.math cimport round
from libc.stdlib cimport malloc, free

from numpy.math cimport INFINITY
cdef double EPSILON = 7./3 - 4./3 - 1

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double max(double[:] x):
    cdef double max = -INFINITY
    for i in range(x.shape[0]):
        if x[i] > max:
            max = x[i]
    return max

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double min(double[:] x):
    cdef double min = INFINITY
    for i in range(x.shape[0]):
        if x[i] < min:
            min = x[i]
    return min

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def ground_template_filter_array(
        double[:, :] input_array,
        double[:] az,
        np.ndarray[np.uint8_t, cast=True, ndim=2] mask,
        double pixel_size,
        bool groundmap=False,
        bool lr=False,
        np.ndarray[np.uint8_t, cast=True, ndim=2] filtmask=None):
    '''
    Remove ground template from array timestreams

    Parameters
    ----------
    input_array: array_like
        shape: (number of channels, number of time steps)
        Input timestream, mutated inplace.
    az: array_like
        shape: input_array[0]
        The azimuth of the timestream.
    mask: array_like
        shape: input_array
        dtype: bool
    pixel_size: float
    groundmap: bool
        If groundmap = True, then do the exact opposite,
        and remove component from timestream that isn't fixed with the ground
    lr: bool
        If true, ground substraction done separately on left and right moving scans
    filtmask: array_like
        shape: input_array
        dtype: bool
        default: None
        filtmask is preprocessed from mask and addtional masking e.g. from point source
        In largepatch filtmask refers to wafermask_chan_filt
    '''
    # initialize
    if filtmask is not None:
        mask = filtmask
    cdef int nCh = input_array.shape[0]
    cdef int nTime = input_array.shape[1]
    cdef double az_min = min(az)
    cdef double az_range = max(az) - az_min
    # Calculate number of pixels given the pixel size
    cdef int nPix = <int>round(az_range / pixel_size)

    # get pointing: an array where each elements is the n-th bin
    # beaware of the (n+1)-th bin that needs to be dealt with separately
    cdef Py_ssize_t* pointing = <Py_ssize_t*>malloc(nTime * sizeof(Py_ssize_t))
#     pointing_init = cvarray(shape=(nTime,), itemsize=sizeof(Py_ssize_t), format="i")
#     cdef Py_ssize_t[:] pointing = pointing_init
    cdef double nPix_per_range = nPix / az_range
    cdef Py_ssize_t i, j
    for i in range(nTime):
        # possible values: [0, 1, ..., nPix]
        pointing[i] = <Py_ssize_t>((az[i] - az_min) * nPix_per_range)

    # bins are arrays of pixels
    cdef int nBin = nPix + 1
    # the signal
#     bins_signal = cvarray(shape=(nCh, nBin), itemsize=sizeof(double), format="i")
#     bins_signal[...] = 0
#     # number of hits of signals
#     bins_hit = cvarray(shape=(nCh, nBin), itemsize=sizeof(int), format="i")
#     bins_hit[...] = 0
    cdef double[:, :] bins_signal = np.zeros((nCh, nBin))
    # number of hits of signals
    # the use of machine epsilon is to avoid special casing 0
    cdef double[:, :] bins_hit = np.full((nCh, nBin), EPSILON)

    cdef Py_ssize_t k
    if not lr:
        # calculate ground template
        # TODO: SIMD
        for i in range(nCh):
            for j in range(nTime):
                if mask[i, j]:
                    k = pointing[j]
                    bins_signal[i, k] += input_array[i, j]
                    bins_hit[i, k] += 1
                    # the following should be better for SIMD, but is slower for me
#                 k = pointing[j]
#                 bins_signal[i, k] += input_array[i, j] * mask[i, j]
#                 bins_hit[i, k] += mask[i, j]
        for i in range(nCh):
            for j in range(nPix - 1):
                bins_signal[i, j] /= bins_hit[i, j]
            # combine last 2 bins to last pixel
            bins_signal[i, nPix - 1] = \
                (bins_signal[i, nPix - 1] + bins_signal[i, nPix]) /\
                (bins_hit[i, nPix - 1] + bins_hit[i, nPix])
        # substraction
        for i in range(nCh):
            for j in range(nTime):
                input_array[i, j] -= bins_signal[i, pointing[j]]

    free(pointing)
