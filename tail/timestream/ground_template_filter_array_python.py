import numpy as np


def simple_map_prepare(nch, npix):
    signal_map = np.zeros((nch, npix))
    hits_map = np.zeros((nch, npix), dtype=np.int32)
    return signal_map, hits_map

# simple_map_python


def simple_map(signal_map, hits_map, pointing, mask, array):
    nch, nt = array.shape
    npix = signal_map.shape[1]

    for i in range(nch):
        for j in range(nt):
            if mask[i, j]:
                signal_map[i, pointing[j]] += array[i, j]
                hits_map[i, pointing[j]] += 1
    ok = hits_map > 0
    signal_map[ok] /= hits_map[ok]

# simple_scan_subtract_python


def simple_scan_subtract(signal_map, pointing, array):
    nch, nt = array.shape
    npix = signal_map.shape[1]
    for i in range(nch):
        for j in range(nt):
            array[i, j] -= signal_map[i, pointing[j]]


def ground_template_filter_array(
        array,
        az,
        mask,
        pixel_size,
        groundmap=False,
        lr=False,
        filtmask=None):
    '''
    Remove ground template from array timestreams

    If groundmap = True, then do the exact opposite, and remove component from timestream that isn't fixed with the ground

    filtmask means to compute the filter template with that subset of the data, operation is applied to data specified by mask

    In largepatch filtmask refers to wafermask_chan_filt
    '''
    nch, nt = array.shape
    minaz = np.min(az)
    maxaz = np.max(az)

    assert mask.shape == array.shape
    assert az.size == nt
    assert 3 * pixel_size < (maxaz - minaz)

    if filtmask is None:
        filtmask = mask

    npix = int(np.round((maxaz - minaz) / pixel_size))
    pixel_size = (maxaz - minaz) / npix

    # bin at npix is used as a junk pixel
    signal_map, hits_map = simple_map_prepare(nch, npix + 1)

    pointing = np.int_(np.floor(npix * (az - minaz) / (maxaz - minaz)))
    # One point with az = maxaz will end up one bin too far left
    pointing[pointing == npix] = npix - 1
    assert np.max(pointing) < npix
    assert np.min(pointing) >= 0

    if groundmap:
        array_in = array.copy()

    if lr:
        vaz = np.gradient(az)
        l = vaz >= 0
        r = vaz < 0
        lp = pointing.copy()
        lp[~l] = npix
        rp = pointing.copy()
        rp[~r] = npix
        simple_map(signal_map, hits_map, lp, filtmask, array)
        signal_map[:, npix] = 0
        simple_scan_subtract(signal_map, lp, array)
        signal_map[:] = 0
        hits_map[:] = 0
        simple_map(signal_map, hits_map, rp, filtmask, array)
        signal_map[:, npix] = 0
        simple_scan_subtract(signal_map, rp, array)
    else:
        simple_map(signal_map, hits_map, pointing, filtmask, array)
        simple_scan_subtract(signal_map, pointing, array)

    if groundmap:
        array[:, :] = array_in - array
