import numpy as np
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.fourier import fourier_gaussian


def bandpass(image, lshort, llong, threshold=1):
    """Convolve with a Gaussian to remove short-wavelength noise,
    and subtract out long-wavelength variations,
    retaining features of intermediate scale."""
    if not 2*lshort < llong:
        raise ValueError("The smoothing length scale must be more" +
                         "than twice the noise length scale.")
    settings = dict(mode='nearest', cval=0)
    boxcar = uniform_filter(image, 2*llong+1, **settings)
    gaussian = np.fft.ifftn(fourier_gaussian(np.fft.fftn(image), lshort))
    result = gaussian - boxcar
    result -= threshold  # Features must be this level above the background.
    return result.real.clip(min=0.)


def scale_to_gamut(image, original_dtype):
    max_value = np.iinfo(original_dtype).max
    scaled = (max_value/image.max()*image.clip(min=0.))
    return scaled.astype(original_dtype)
