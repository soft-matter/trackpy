import logging

import numpy as np
from scipy.ndimage import uniform_filter1d, correlate1d, fourier_gaussian

from .utils import validate_tuple
from .masks import gaussian_kernel
try:
    from pims import pipeline
except ImportError:
    pipeline = lambda x: x

def lowpass(image, sigma=1, truncate=4):
    """Remove noise by convolving with a Gaussian.

    Convolve with a Gaussian to remove short-wavelength noise.

    The lowpass implementation relies on scipy.ndimage.filters.gaussian_filter,
    and it is the fastest way known to the authors of performing a bandpass in
    Python.

    Parameters
    ----------
    image : ndarray
    sigma : number or tuple, optional
        Size of the gaussian kernel with which the image is convolved.
        Provide a tuple for different sizes per dimension. Default 1.
    truncate : number, optional
        Determines the truncation size of the convolution kernel. Default 4.

    Returns
    -------
    result : array
        the processed image, as float

    See Also
    --------
    bandpass
    """
    sigma = validate_tuple(sigma, image.ndim)
    result = np.array(image, dtype=float)
    for axis, _sigma in enumerate(sigma):
        if _sigma > 0:
            correlate1d(result, gaussian_kernel(_sigma, truncate), axis,
                        output=result, mode='constant', cval=0.0)
    return result


def boxcar(image, size):
    """Compute a rolling (boxcar) average of an image.

    The kernel is square or rectangular.

    Parameters
    ----------
    image : ndarray
    size : number or tuple
        Size of rolling average (square or rectangular kernel) filter. Should
        be odd and larger than the particle diameter.
        Provide a tuple for different sizes per dimension.

    Returns
    -------
    result : array
        the rolling average image

    See Also
    --------
    bandpass
    """
    size = validate_tuple(size, image.ndim)
    if not np.all([x & 1 for x in size]):
        raise ValueError("Smoothing size must be an odd integer. Round up.")
    result = image.copy()
    for axis, _size in enumerate(size):
        if _size > 1:
            uniform_filter1d(result, _size, axis, output=result,
                             mode='nearest', cval=0)
    return result


def bandpass(image, lshort, llong, threshold=None, truncate=4):
    """Remove noise and background variation.

    Convolve with a Gaussian to remove short-wavelength noise and subtract out
    long-wavelength variations by subtracting a running average. This retains
    features of intermediate scale.

    The lowpass implementation relies on scipy.ndimage.filters.gaussian_filter,
    and it is the fastest way known to the authors of performing a bandpass in
    Python.

    Parameters
    ----------
    image : ndarray
    lshort : number or tuple
        Size of the gaussian kernel with which the image is convolved.
        Provide a tuple for different sizes per dimension.
    llong : integer or tuple
        The size of rolling average (square or rectangular kernel) filter.
        Should be odd and larger than the particle diameter.
        When llong <= lshort, an error is raised.
        Provide a tuple for different sizes per dimension.
    threshold : float or integer
        Clip bandpass result below this value. Thresholding is done on the
        already background-subtracted image.
        By default, 1 for integer images and 1/255 for float images.
    truncate : number, optional
        Determines the truncation size of the gaussian kernel. Default 4.

    Returns
    -------
    result : array
        the bandpassed image

    See Also
    --------
    lowpass, boxcar, legacy_bandpass, legacy_bandpass_fftw

    Notes
    -----
    The boxcar size and shape changed in v0.4: before, the boxcar had a
    circular kernel with radius `llong`, now it is has a square kernel that
    has an edge length of `llong` (twice as small!).
    """
    lshort = validate_tuple(lshort, image.ndim)
    llong = validate_tuple(llong, image.ndim)
    if np.any([x >= y for (x, y) in zip(lshort, llong)]):
        raise ValueError("The smoothing length scale must be larger than " +
                         "the noise length scale.")
    if threshold is None:
        if np.issubdtype(image.dtype, np.integer):
            threshold = 1
        else:
            threshold = 1/255.
    background = boxcar(image, llong)
    result = lowpass(image, lshort, truncate)
    result -= background
    return np.where(result >= threshold, result, 0)

@pipeline
def invert_image(raw_image, max_value=None):
    """Invert the image.

    Use this to convert dark features on a bright background to bright features
    on a dark background.

    Parameters
    ----------
    image : ndarray
    max_value : number
        The maximum value of the image. Optional. If not given, this will is
        (for integers) the highest possible value and (for floats) 1.

    Returns
    -------
    inverted image
    """
    if max_value is None:
        if np.issubdtype(raw_image.dtype, np.integer):
            max_value = np.iinfo(raw_image.dtype).max
        else:
            # To avoid degrading performance, assume gamut is zero to one.
            # Have you ever encountered an image of unnormalized floats?
            max_value = 1.

    # It is tempting to do this in place, but if it is called multiple
    # times on the same image, chaos reigns.
    if np.issubdtype(raw_image.dtype, np.integer):
        result = raw_image ^ max_value
    else:
        result = max_value - raw_image
    return result


def convert_to_int(image, dtype='uint8'):
    """Convert the image to integer and normalize if applicable.

    Clips all negative values to 0. Does nothing if the image is already
    of integer type.

    Parameters
    ----------
    image : ndarray
    dtype : numpy dtype
        dtype to convert to. If the image is already of integer type, this
        argument is ignored. Must be integer-subdtype. Default 'uint8'.

    Returns
    -------
    tuple of (scale_factor, image)
    """
    if np.issubdtype(image.dtype, np.integer):
        # Do nothing, image is already of integer type.
        return 1., image
    max_value = np.iinfo(dtype).max
    image_max = image.max()
    if image_max == 0:  # protect against division by zero
        scale_factor = 1.
    else:
        scale_factor = max_value / image_max
    return scale_factor, (scale_factor * image.clip(min=0.)).astype(dtype)


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
