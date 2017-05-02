import numpy as np
import weave


def simple_map_prepare(nCh, nPix):
    """
    Prepare a simple map.

    Parameters
    ----------
    nCh: int
        Number of channels
    nPix: int
        Number of pixels

    Returns
    -------
    signal_map: ndarray
        an array of length nCh, where each entry is a length-nPix zero array.
    hits_map: ndarray
        same as above with dtype int32
    """
    signal_map = np.zeros((nCh, nPix))
    hits_map = np.zeros((nCh, nPix), dtype=np.int32)
    return signal_map, hits_map


def simple_map(signal_map, hits_map, pointing, mask, array):
    ''' Map timestream into a signal map '''
    nch, nt = array.shape
    npix = signal_map.shape[1]
    assert signal_map.dtype == array.dtype
    assert mask.dtype == np.bool

    c_code = '''
	int ch,t;
	int pix;
	for(ch=0;ch<nch;ch++) {
		for(t=0;t<nt;t++) {
			pix = pointing(t);
			signal_map(ch,pix) += array(ch,t)*mask(ch,t);
			hits_map(ch,pix) += mask(ch,t);
		}
	}

	for(ch=0;ch<nch;ch++) {
		for(pix=0;pix<npix;pix++) {
			if(hits_map(ch,pix) > 0) {
				signal_map(ch,pix) /= hits_map(ch,pix);
			}
		}
	}
	'''

    weave.inline(c_code, ['array', 'pointing', 'mask', 'signal_map', 'hits_map',
                          'nch', 'nt', 'npix'], type_converters=weave.converters.blitz)


def simple_scan_subtract(signal_map, pointing, array):
    ''' Directly subtract signal map from array '''
    nch, nt = array.shape
    npix = signal_map.shape[1]

    assert np.max(pointing) < npix
    assert np.min(pointing) >= 0

    c_code = '''
	int ch,t;
	int pix;
	for(ch=0;ch<nch;ch++) {
		for(t=0;t<nt;t++) {
			pix = pointing(t);
			array(ch,t) -= signal_map(ch,pix);
		}
	}
	'''

    weave.inline(c_code, ['array', 'pointing', 'npix', 'nt', 'nch',
                          'signal_map'], type_converters=weave.converters.blitz)


def ground_template_filter_array(
        input_array,
        az,
        mask,
        pixel_size,
        groundmap=False,
        lr=False,
        filtmask=None):
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
        filtmask means to compute the filter template with that subset of the data,
        operation is applied to data specified by mask
        In largepatch filtmask refers to wafermask_chan_filt
    '''
    # initialize
    nCh, nTime = input_array.shape
    az_min = np.min(az)
    az_max = np.max(az)
    az_range = az_max - az_min
    if filtmask is None:
        filtmask = mask

    assert input_array.shape == mask.shape == filtmask.shape
    assert az.size == nTime

    # Calculate number of pixels given the pixel size
    nPix = int(np.round(az_range / pixel_size))
    assert nPix > 3
    # recalculate pixel_size (because nPix is int)
    pixel_size = az_range / nPix

    # bin at nPix is used as a junk pixel
    signal_map, hits_map = simple_map_prepare(nCh, nPix + 1)

    # get pointing
    pointing = np.int_(np.floor(nPix * (az - az_min) / az_range))
    # One point with az = az_max will end up one bin too far left
    pointing[pointing == nPix] = nPix - 1

    if groundmap:
        array_in = input_array.copy()

    if lr:
        vaz = np.gradient(az)
        # select left / right moving timestream
        l = vaz >= 0
        r = vaz < 0
        pointingL = pointing.copy()
        pointingL[~l] = nPix
        pointingR = pointing.copy()
        pointingR[~r] = nPix

        simple_map(signal_map, hits_map, pointingL, filtmask, input_array)
        signal_map[:, nPix] = 0
        simple_scan_subtract(signal_map, pointingL, input_array)
        signal_map[:] = 0
        hits_map[:] = 0
        simple_map(signal_map, hits_map, pointingR, filtmask, input_array)
        signal_map[:, nPix] = 0
        simple_scan_subtract(signal_map, pointingR, input_array)
    else:
        simple_map(signal_map, hits_map, pointing, filtmask, input_array)
        simple_scan_subtract(signal_map, pointing, input_array)

    if groundmap:
        input_array[:, :] = array_in - input_array
