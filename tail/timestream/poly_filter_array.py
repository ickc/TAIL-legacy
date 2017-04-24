import numpy as np


class LegCache():
    def __init__(self):
        self.d = {}
        pass

    def prep_legendre(self, n, polyorder):
        p = (n, polyorder)
        if p not in self.d:
            self.d[p] = prep_legendre(n, polyorder)
        return self.d[p]


def prep_legendre(n, polyorder):
    '''make array of legendre's'''
    legendres = np.empty([n, polyorder + 1])
    legendres[:, 0] = np.ones(n)
    if polyorder > 0:
        legendres[:, 1] = np.linspace(-1, 1, n)
    for i in range(polyorder - 1):
        l = i + 1
        legendres[:,
                  l + 1] = ((2 * l + 1) * legendres[:,
                                                    1] * legendres[:,
                                                                   l] - l * legendres[:,
                                                                                      l - 1]) / (l + 1)

    q, r = np.linalg.qr(legendres)
    rinv = np.linalg.inv(r)
    qt = q.T.copy()
    return legendres, rinv, qt


def poly_filter_array(
        array,
        mask_remove,
        mask,
        scan_list,
        ibegin,
        polyorder,
        minfrac=.75):
    """ writes over input array
    """
    nold = -1
    # do nothing
    if polyorder < 0:
        return array
    #damn, work
    nch = array.shape[0]
    nt = array.shape[1]
    ns = len(scan_list)

    coeff_out = np.zeros((nch, ns, polyorder + 1))

    legcache = LegCache()

    # remove mean
    if polyorder == 0:
        for s in range(len(scan_list)):
            istart, n = scan_list[s]
            start = istart - ibegin
#			array[:,start:start+n] -= tile(np.mean(array[:,start:start+n]*mask[:,start:start+n],axis=1).reshape(nch,1),[1,n])
            for i in range(nch):
                if np.any(mask[i, start:start + n]):
                    mean = np.average(
                        array[i, start:start + n], weights=mask[i, start:start + n])
                    array[i, start:start + n] -= mean
                    coeff_out[i, s, 0] = mean

    # other cases
    if polyorder > 0:
        for s in range(len(scan_list)):
            istart, n = scan_list[s]
            start = istart - ibegin
            if n <= polyorder:  # otherwise cannot compute legendre polynomials
                for i in xrange(nch):
                    mask[i, start:start + n] = 0  # flag it
                    # remove this region from actual data as well
                    mask_remove[i, start:start + n] = 0
                    print 'Not enough points (%d) to build legendre of order (%d)' % (n, polyorder)
                continue
            goodhits = np.sum(mask[:, start:start + n], axis=1)
            if n != nold:
                legendres, rinv, qt = legcache.prep_legendre(n, polyorder)
                rinvqt = np.dot(rinv, qt)
                nold = n
            # handle no masked ones

            for i in xrange(nch):
                if goodhits[i] != n:
                    continue  # skip for now
                # filter_slice_legendre_qr_nomask_precalc_inplace(array[i,start:start+n],legendres,rinvqt)
                bolo = array[i, start:start + n]
                coeff = np.dot(rinvqt, bolo)
                coeff_out[i, s, :] = coeff
                bolo -= np.dot(legendres, coeff)
#				array[i,start:start+n] = filter_slice_legendre_qr_nomask_precalc(
#					array[i,start:start+n], legendres,rinv,qt)

            for i in xrange(nch):
                if goodhits[i] == n:
                    continue  # skip since dealt with above
                if goodhits[i] < minfrac * n:  # not enough points
                    mask[i, start:start + n] = 0  # flag it
                    # remove this region from actual data as well
                    mask_remove[i, start:start + n] = 0
                    continue
                bolo, coeff = filter_slice_legendre_qr_mask_precalc(
                    array[i, start:start + n], mask[i, start:start + n], legendres)
                array[i, start:start + n] = bolo
                coeff_out[i, s, :] = coeff
    return coeff_out
