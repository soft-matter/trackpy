from mr.core.utils import memo
import numpy as np
from scipy.ndimage import filters
from scipy.ndimage import fourier

def bandpass(image, lshort, llong):
    """Convolve with a Gaussian to remove short-wavelength noise,
    and subtract out long-wavelength variations,
    retaining features of intermediate scale."""
    if not 2*lshort < llong:
        raise ValueError, ("The smoothing length scale must be more" 
                           "than twice the noise length scale.")
    smoothed_background = filters.uniform_filter(image, 2*llong+1)
    result = np.fft.ifft2(fourier.fourier_gaussian(np.fft.fft2(image), lshort))
    result -= smoothed_background
    result -= 1 # Features must be at least 1 ADU above the background.
    return result.real.clip(min=0.)

@memo
def circular_mask(diameter, side_length=None):
    """A circle of 1's inscribed in a square of 0's,
    the 'footprint' of the features we seek."""
    r = int(diameter)/2
    L = side_length if side_length else int(diameter)
    points = np.arange(-int(L/2), int(L/2) + 1)
    x, y = np.meshgrid(points, points)
    z = np.sqrt(x**2 + y**2)
    return z <= r
