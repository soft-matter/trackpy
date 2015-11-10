from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import logging

import numpy as np
from scipy.ndimage.filters import uniform_filter1d, correlate1d
from scipy.ndimage.fourier import fourier_gaussian

from .utils import validate_tuple
from .masks import gaussian_kernel


def bandpass(image, lshort, llong, threshold=None, truncate=4):
    """Remove noise and background variation.

    Convolve with a Gaussian to remove short-wavelength noise and subtract out
    long-wavelength variations, retaining features of intermediate scale.

    This implementation relies on scipy.ndimage.filters.gaussian_filter, and it
    is the fastest way known to the authors of performing a bandpass in
    Python.

    Parameters
    ----------
    image : ndarray
    lshort : small-scale cutoff (noise)
    llong : large-scale cutoff
    for both lshort and llong:
        give a tuple value for different sizes per dimension
        give int value for same value for all dimensions
        when 2*lshort >= llong, no noise filtering is applied
    threshold : float or integer
        By default, 1 for integer images and 1/256. for float images.

    Returns
    -------
    result : array
        the bandpassed image

    See Also
    --------
    legacy_bandpass, legacy_bandpass_fftw
    """
    lshort = validate_tuple(lshort, image.ndim)
    llong = validate_tuple(llong, image.ndim)
    if np.any([x*2 >= y for (x, y) in zip(lshort, llong)]):
        raise ValueError("The smoothing length scale must be more" +
                         "than twice the noise length scale.")
    if threshold is None:
        if np.issubdtype(image.dtype, np.integer):
            threshold = 1
        else:
            threshold = 1/256.
    boxcar = image.copy()
    result = np.array(image, dtype=np.float)
    for axis, (sigma, smoothing) in enumerate(zip(lshort, llong)):
        if smoothing > 1:
            uniform_filter1d(boxcar, 2*smoothing+1, axis, output=boxcar,
                             mode='nearest', cval=0)
        if sigma > 0:
            correlate1d(result, gaussian_kernel(sigma, truncate), axis,
                        output=result, mode='constant', cval=0.0)
    result -= boxcar
    return np.where(result > threshold, result, 0)


# Below are two older implementations of bandpass. Formerly, they were lumped
# into one function, ``bandpass``, that used pyfftw if it was available and
# numpy otherwise. Now there are separate functions for the pyfftw and numpy
# code paths.

# Both of these have been found to be slower than the new ``bandpass`` above
# when benchmarked on typical inputs. Nonetheless, they are retained in case
# they offer some advantage unforeseen by the authors.

# All three functions give identical results, up to small numerical errors.


logger = logging.getLogger(__name__)

try:
    import pyfftw
except ImportError:
    # Use numpy.
    FFTW_AVAILABLE = False
else:
    FFTW_AVAILABLE = True
    pyfftw.interfaces.cache.enable()
    planned = False

    def fftn(a):
        global planned
        if not planned:
            logger.info("Note: FFTW is configuring itself. This will take " +
                        "several seconds, but subsequent calls will run " +
                        "*much* faster.")
            planned = True
        a = pyfftw.n_byte_align(a, a.dtype.alignment)
        return pyfftw.interfaces.numpy_fft.fftn(a).astype(np.complex128)

    def ifftn(a):
        a = pyfftw.n_byte_align(a, a.dtype.alignment)
        return pyfftw.interfaces.numpy_fft.ifftn(a)


def legacy_bandpass(image, lshort, llong, threshold=None):
    """Remove noise and background variation.

    Convolve with a Gaussian to remove short-wavelength noise and subtract out
    long-wavelength variations, retaining features of intermediate scale.

    This implementation performs a Fourier transform using numpy.
    In benchmarks using typical inputs, it was found to be slower than the
    ``bandpass`` function in this module.

    Parameters
    ----------
    image : ndarray
    lshort : small-scale cutoff (noise)
    llong : large-scale cutoff
    for both lshort and llong:
        give a tuple value for different sizes per dimension
        give int value for same value for all dimensions
        when 2*lshort >= llong, no noise filtering is applied
    threshold : float or integer
        By default, 1 for integer images and 1/256. for float images.

    Returns
    -------
    result : array
        the bandpassed image

    See Also
    --------
    bandpass, legacy_bandpass_fftw
    """
    fftn = np.fft.fftn
    ifftn = np.fft.ifftn
    lshort = validate_tuple(lshort, image.ndim)
    llong = validate_tuple(llong, image.ndim)
    if np.any([x*2 >= y for (x, y) in zip(lshort, llong)]):
        raise ValueError("The smoothing length scale must be more" +
                         "than twice the noise length scale.")
    if threshold is None:
        if np.issubdtype(image.dtype, np.integer):
            threshold = 1
        else:
            threshold = 1/256.
    # Perform a rolling average (boxcar) with kernel size = 2*llong + 1
    boxcar = np.asarray(image)
    for (axis, size) in enumerate(llong):
        boxcar = uniform_filter1d(boxcar, size*2+1, axis, mode='nearest',
                                  cval=0)
    # Perform a gaussian filter
    gaussian = ifftn(fourier_gaussian(fftn(image), lshort)).real

    result = gaussian - boxcar
    return np.where(result > threshold, result, 0)


def legacy_bandpass_fftw(image, lshort, llong, threshold=None):
    """Remove noise and background variation.

    Convolve with a Gaussian to remove short-wavelength noise and subtract out
    long-wavelength variations, retaining features of intermediate scale.

    This implementation performs a Fourier transform using FFTW
    (Fastest Fourier Transform in the West). Without FFTW and pyfftw, it
    will raise an ImportError

    In benchmarks using typical inputs, it was found to be slower than the
    ``bandpass`` function in this module.

    Parameters
    ----------
    image : ndarray
    lshort : small-scale cutoff (noise)
    llong : large-scale cutoff
    for both lshort and llong:
        give a tuple value for different sizes per dimension
        give int value for same value for all dimensions
        when 2*lshort >= llong, no noise filtering is applied
    threshold : float or integer
        By default, 1 for integer images and 1/256. for float images.

    Returns
    -------
    result : array
        the bandpassed image

    See Also
    --------
    bandpass, legacy_bandpass
    """
    if not FFTW_AVAILABLE:
        raise ImportError("This implementation requires pyfftw.")
    lshort = validate_tuple(lshort, image.ndim)
    llong = validate_tuple(llong, image.ndim)
    if np.any([x*2 >= y for (x, y) in zip(lshort, llong)]):
        raise ValueError("The smoothing length scale must be more" +
                         "than twice the noise length scale.")
    if threshold is None:
        if np.issubdtype(image.dtype, np.integer):
            threshold = 1
        else:
            threshold = 1/256.
    # Perform a rolling average (boxcar) with kernel size = 2*llong + 1
    boxcar = np.asarray(image)
    for (axis, size) in enumerate(llong):
        boxcar = uniform_filter1d(boxcar, size*2+1, axis, mode='nearest',
                                  cval=0)
    # Perform a gaussian filter
    gaussian = ifftn(fourier_gaussian(fftn(image), lshort)).real

    result = gaussian - boxcar
    return np.where(result > threshold, result, 0)


def scalefactor_to_gamut(image, original_dtype):
    return np.iinfo(original_dtype).max / image.max()


def scale_to_gamut(image, original_dtype, scale_factor=None):
    if scale_factor is None:
        scale_factor = scalefactor_to_gamut(image, original_dtype)
    scaled = (scale_factor * image.clip(min=0.)).astype(original_dtype)
    return scaled
