import numpy as np
cimport numpy as np
cimport cython

from cython.parallel cimport prange

class LegCache(object):
    def __init__(self):
        self.d = {}
        pass

    def prep_legendre(self, int n, int polyorder):
        p = (n, polyorder)
        if p not in self.d:
            self.d[p] = prep_legendre(n, polyorder)
        return self.d[p]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline prep_legendre(int n, int polyorder):
    '''make array of legendre's'''
    cdef np.ndarray[np.float64_t, ndim=2] legendres = np.empty([n, polyorder + 1])
    legendres[:, 0] = np.ones(n)
    if polyorder > 0:
        legendres[:, 1] = np.linspace(-1, 1, n)
    cdef Py_ssize_t i, l
    for i in range(polyorder - 1):
        l = i + 1
        legendres[:,
                  l + 1] = ((2 * l + 1) * legendres[:,
                                                    1] * legendres[:,
                                                                   l] - l * legendres[:,
                                                                                      l - 1]) / (l + 1)
    cdef np.float64_t[:, :] q, r, rinv, qt
    q, r = np.linalg.qr(legendres)
    rinv = np.linalg.inv(r)
    qt = q.T.copy()
    return legendres, rinv, qt

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline filter_slice_legendre_qr_mask_precalc(np.float64_t[:] bolo, np.float64_t[:] mask, legendres):
    cdef int m = legendres.shape[1]
    cdef int n = legendres.shape[0]
    cdef np.float64_t[:, :] l2, q, r, rinv, p, coeff, out
    l2 = legendres * np.tile(mask.reshape(n, 1), [1, m])
    q, r = np.linalg.qr(l2)

    rinv = np.linalg.inv(r)
    p = np.dot(q.T,bolo)
    coeff = np.dot(rinv, p)
    out = bolo - np.dot(legendres, coeff)
    return out, coeff

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def poly_filter_array(
        np.ndarray[np.float64_t, ndim=2] input_array,
        np.ndarray[np.uint8_t, cast=True, ndim=2] mask_remove,
        np.ndarray[np.uint8_t, cast=True, ndim=2] mask,
        scan_list,
        int ibegin,
        int polyorder,
        double minfrac=.75,
        int num_threads=4):
    """
    Parameters
    ----------
    input_array: numpy.ndarray
        dtype: float64
        shape: (number of channels, number of time steps)
        Input timestream, mutated inplace.
    mask_remove: numpy.ndarray
        dtype: bool
        shape: same as input_array
    mask: numpy.ndarray
        dtype: bool
        shape: same as input_array
        may be the same as mask_remove
    scan_list: array_like
        dtype: numpy.int64
        shape: (number of scan, 2)
        each element contains (starting point of the scan, length of the scan)
    ibegin: int
    polyorder: int
    minfrac: float

    Returns
    -------
    coeff_out: numpy.ndarray
        dtype: float64
        shape: (input_array[0], len(scan_list), polyorder + 1)
    """
    cdef np.int64_t [:, :] scan_view = np.array(scan_list)

    cdef int nold = -1
    # do nothing
    if polyorder < 0:
        return None

    cdef Py_ssize_t nCh, nt, ns
    nch = input_array.shape[0]
    nt = input_array.shape[1]
    ns = scan_view.shape[0]

    cdef np.ndarray[np.float64_t, ndim=3] coeff_out = np.zeros((nch, ns, polyorder + 1))

    legcache = LegCache()

    cdef Py_ssize_t s, i, j
    cdef Py_ssize_t istart, n, start

    cdef double mean
    cdef np.int64_t[:] goodhits

    # remove mean
    if polyorder == 0:
        for s in range(ns):
            istart = scan_view[s][0]
            n = scan_view[s][1]
            start = istart - ibegin
            for i in range(nch):
                if np.any(mask[i, start:start + n]):
                    mean = np.average(
                        input_array[i, start:start + n], weights=mask[i, start:start + n])
                    for j in prange(start, start + n, nogil=True, schedule='guided', num_threads=num_threads):
                        input_array[i, j] -= mean
                    coeff_out[i, s, 0] = mean

    # other cases
    if polyorder > 0:
        for s in range(ns):
            istart = scan_view[s][0]
            n = scan_view[s][1]
            start = istart - ibegin
            if n <= polyorder:  # otherwise cannot compute legendre polynomials
                for i in prange(nch, nogil=True, schedule='guided', num_threads=num_threads):
                    for j in prange(start, start + n):
                        mask[i, j] = False  # flag it
                    # remove this region from actual data as well
                    for j in prange(start, start + n):
                        mask_remove[i, j] = False
                print('Not enough points (%d) to build legendre of order (%d)' % (n, polyorder))
                continue
            goodhits = np.sum(mask[:, start:start + n], axis=1)
            if n != nold:
                legendres, rinv, qt = legcache.prep_legendre(n, polyorder)
                rinvqt = np.dot(rinv, qt)
                nold = n
            # handle no masked ones

            for i in range(nch):
                if goodhits[i] != n:
                    continue  # skip for now
                coeff = np.dot(rinvqt, input_array[i, start:start + n])
                coeff_out[i, s, :] = coeff
                input_array[i, start:start + n] -= np.dot(legendres, coeff)
#                input_array[i,start:start+n] = filter_slice_legendre_qr_nomask_precalc(
#                    input_array[i,start:start+n], legendres,rinv,qt)

            for i in range(nch):
                if goodhits[i] == n:
                    continue  # skip since dealt with above
                if goodhits[i] < minfrac * n:  # not enough points
                    mask[i, start:start + n] = False  # flag it
                    # remove this region from actual data as well
                    mask_remove[i, start:start + n] = False
                    continue
                input_array[i, start:start + n], coeff = filter_slice_legendre_qr_mask_precalc(
                    input_array[i, start:start + n], mask[i, start:start + n], legendres)
                coeff_out[i, s, :] = coeff
    return coeff_out
