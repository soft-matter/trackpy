import sys
import warnings
import numpy as np
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.fourier import fourier_gaussian


# When loading module, try to use pyFFTW ("Fastest Fourier Transform in the
# West") if it is available.
try:
    import pyfftw
except ImportError:
    # Use numpy.
    USING_FFTW = False
    fftn = np.fft.fftn
    ifftn = np.fft.ifftn
else:
    USING_FFTW = True
    pyfftw.interfaces.cache.enable()
    planned = False
    def _maybe_align(a):
        global planned
        if not planned:
            warnings.warn("FFTW is configuring itself. This will take " +
                          "several sections, but subsequent calls will run " +
                          "*much* faster.", UserWarning)
            sys.stderr.flush()  # Show that warning immediately.
            planned = True 
        return pyfftw.n_byte_align(a, a.dtype.alignment)
    fftn = lambda a: pyfftw.interfaces.numpy_fft.fftn(_maybe_align(a))
    ifftn = lambda a: pyfftw.interfaces.numpy_fft.ifftn(_maybe_align(a))


def bandpass(image, lshort, llong, threshold=1):
    """Convolve with a Gaussian to remove short-wavelength noise,
    and subtract out long-wavelength variations,
    retaining features of intermediate scale."""
    if not 2*lshort < llong:
        raise ValueError("The smoothing length scale must be more" +
                         "than twice the noise length scale.")
    settings = dict(mode='nearest', cval=0)
    boxcar = uniform_filter(image, 2*llong+1, **settings)
    gaussian = ifftn(fourier_gaussian(fftn(image), lshort))
    result = gaussian - boxcar
    return np.where(result > threshold, result.real, 0)


def scale_to_gamut(image, original_dtype):
    max_value = np.iinfo(original_dtype).max
    scaled = (max_value/image.max()*image.clip(min=0.))
    return scaled.astype(original_dtype)
