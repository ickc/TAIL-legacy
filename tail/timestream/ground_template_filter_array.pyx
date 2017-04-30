import numpy as np
cimport numpy as np
import cython

from cython.view cimport array as cvarray
from cython.parallel import parallel, prange

from libc.math cimport round
from libc.stdlib cimport malloc, calloc, free
from libcpp cimport bool
from libc.string cimport memset

from numpy.math cimport INFINITY

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
cdef inline _ground_filter(
        double[:, :] input_array,
        np.ndarray[np.uint8_t, cast=True, ndim=2] mask,
        bool groundmap,
        int num_threads,
        int nCh,
        int nTime,
        int nPix,
        Py_ssize_t* pointing):
    cdef Py_ssize_t i, j, k
    # the signal
    # cdef double[:, :] bins_signal = np.zeros((nCh, nBin))
    # last bin correspond to the az_max pixel, just in case it hasn't been masked
    cdef int nBin = nPix + 1
    cdef int size = nCh * nBin
    cdef double* bins_signal = <double*>calloc(size, sizeof(double))
    # number of hits of signals
    # the use of machine epsilon is to avoid special casing 0
    cdef int* bins_hit = <int*>calloc(size, sizeof(double))

    for i in prange(nCh, nogil=True, schedule='guided', num_threads=num_threads):
        # calculate ground template
        ## add total signal and no. of hits
        for j in range(nTime):
            if mask[i, j]:
                k = pointing[j]
                bins_signal[nBin * i + k] += input_array[i, j]
                bins_hit[nBin * i + k] += 1
            # the following should be better for SIMD, but is actually slower
            # k = pointing[j]
            # bins_signal[nBin * i + k] += input_array[i, j] * mask[i, j]
            # bins_hit[nBin * i + k] += mask[i, j]

        ## average signal
        for k in range(nPix):
            if bins_hit[nBin * i + k] != 0:
                bins_signal[nBin * i + k] /= bins_hit[nBin * i + k]

        # substraction
        if groundmap:
            for j in range(nTime):
                input_array[i, j] = bins_signal[nBin * i + pointing[j]]
        else:
            for j in range(nTime):
                input_array[i, j] -= bins_signal[nBin * i + pointing[j]]
    free(bins_signal)
    free(bins_hit)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline _ground_filter_lr(
        double[:, :] input_array,
        double[:] az,
        np.ndarray[np.uint8_t, cast=True, ndim=2] mask,
        bool groundmap,
        int num_threads,
        int nCh,
        int nTime,
        int nPix,
        Py_ssize_t* pointing):
    cdef Py_ssize_t i, j, k
    # the signal
    # cdef double[:, :] bins_signal = np.zeros((nCh, nBin))
    # last bin correspond to the az_max pixel, just in case it hasn't been masked
    cdef int nBin = nPix + 1
    cdef int size = nCh * nBin
    cdef double* bins_signal_l = <double*>calloc(size, sizeof(double))
    cdef double* bins_signal_r = <double*>calloc(size, sizeof(double))
    # number of hits of signals
    # the use of machine epsilon is to avoid special casing 0
    cdef int* bins_hit_l = <int*>calloc(size, sizeof(double))
    cdef int* bins_hit_r = <int*>calloc(size, sizeof(double))
    # is the input_array right moving?
    cdef bool* isMovingRight = <bool*>calloc(nTime, sizeof(bool))

    # detect left vs. right
    for j in prange(nTime - 1, nogil=True, schedule='guided', num_threads=num_threads):
        isMovingRight[j] = az[j + 1] > az[j]
    # the last one does not have a diff.
    isMovingRight[nTime - 1] = isMovingRight[nTime - 2]

    for i in prange(nCh, nogil=True, schedule='guided', num_threads=num_threads):
        # calculate ground template
        ## add total signal and no. of hits
        for j in range(nTime):
            if mask[i, j]:
                k = pointing[j]
                if isMovingRight[j]:
                    bins_signal_r[nBin * i + k] += input_array[i, j]
                    bins_hit_r[nBin * i + k] += 1
                else:
                    bins_signal_l[nBin * i + k] += input_array[i, j]
                    bins_hit_l[nBin * i + k] += 1

        ## average signal
        for k in range(nPix):
            if bins_hit_l[nBin * i + k] != 0:
                bins_signal_l[nBin * i + k] /= bins_hit_l[nBin * i + k]
        for k in range(nPix):
            if bins_hit_r[nBin * i + k] != 0:
                bins_signal_r[nBin * i + k] /= bins_hit_r[nBin * i + k]

        # substraction
        if groundmap:
            for j in range(nTime):
                if isMovingRight[j]:
                    input_array[i, j] = bins_signal_r[nBin * i + pointing[j]]
                else:
                    input_array[i, j] = bins_signal_l[nBin * i + pointing[j]]
        else:
            for j in range(nTime):
                if isMovingRight[j]:
                    input_array[i, j] -= bins_signal_r[nBin * i + pointing[j]]
                else:
                    input_array[i, j] -= bins_signal_l[nBin * i + pointing[j]]
    free(bins_signal_l)
    free(bins_signal_r)
    free(bins_hit_l)
    free(bins_hit_r)
    free(isMovingRight)


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
        np.ndarray[np.uint8_t, cast=True, ndim=2] filtmask=None,
        int num_threads=4):
    '''
    Remove ground template from array timestreams

    Parameters
    ----------
    input_array: numpy.ndarray
        shape: (number of channels, number of time steps)
        Input timestream, mutated inplace.
    az: numpy.ndarray
        shape: input_array[0]
        The azimuth of the timestream.
    mask: numpy.ndarray
        shape: input_array
        dtype: bool
    pixel_size: float
    groundmap: bool
        If groundmap = True, then do the exact opposite,
        and remove component from timestream that isn't fixed with the ground
    lr: bool
        If true, ground substraction done separately on left and right moving scans
    filtmask: numpy.ndarray
        shape: input_array
        dtype: bool
        default: None
        filtmask is preprocessed from mask and addtional masking e.g. from point source
        In largepatch filtmask refers to wafermask_chan_filt
    '''
    # loop indices
    cdef Py_ssize_t i

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
    # beaware of the last bin that will always be masked away because it's a turning point
    cdef Py_ssize_t* pointing = <Py_ssize_t*>malloc(nTime * sizeof(Py_ssize_t))
    cdef double nPix_per_range = nPix / az_range
    for i in prange(nTime, nogil=True, schedule='guided', num_threads=num_threads, num_threads=num_threads):
        # possible values: [0, 1, ..., nPix]
        pointing[i] = <Py_ssize_t>((az[i] - az_min) * nPix_per_range)

    if lr:
        _ground_filter_lr(
                input_array,
                az,
                mask,
                groundmap,
                num_threads,
                nCh,
                nTime,
                nPix,
                pointing)
    else:
        _ground_filter(
                input_array,
                mask,
                groundmap,
                num_threads,
                nCh,
                nTime,
                nPix,
                pointing)

    free(pointing)
