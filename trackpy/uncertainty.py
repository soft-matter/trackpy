from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np
from scipy.ndimage import morphology

from .preprocessing import bandpass
from .masks import binary_mask, N_binary_mask, root_sum_x_squared
from .utils import validate_tuple


def roi(image, diameter, threshold=None):
    """Return a mask selecting the neighborhoods of bright regions.
    See Biophysical journal 88(1) 623-638 Figure C.

    Parameters
    ----------
    image : ndarray
    diameter : feature size used for centroid identification

    Returns
    -------
    boolean ndarray, True around bright regions
    """
    diameter = validate_tuple(diameter, image.ndim)
    signal_mask = bandpass(image, 1, tuple([d + 1 for d in diameter]), threshold)
    radius = tuple([int(d)//2 for d in diameter])
    structure = binary_mask(radius, image.ndim)
    signal_mask = morphology.binary_dilation(signal_mask, structure=structure)
    return signal_mask


def measure_noise(image, diameter, threshold):
    "Compute the standard deviation of the dark pixels outside the signal."
    signal_mask = roi(image, diameter, threshold)
    return image[~signal_mask].mean(), image[~signal_mask].std()


def static_error(mass, black_level, noise, radius, noise_size=1):
    """Compute the uncertainty in particle position ("the static error").

    Parameters
    ----------
    mass : ndarray of feature masses
    black_level : number, average background pixel value
    noise : number, standard deviation of the noise
    radius : tuple of feature radius used to locate centroids
    noise_size : noise correlation length, may be tuple-valued (?)

    Returns
    -------
    Series of static error estimates, indexed like the trajectories.
    When either radius or noise_size are anisotropic, a list of ndim Series is
    returned, one for each dimension. The order of this list is equal to the
    order of coord_columns and the reverse of radius and noise_size.

    Note
    ----
    This is an adjusted version of the process described by Thierry Savin and
    Patrick S. Doyle in their paper "Static and Dynamic Errors in Particle
    Tracking Microrheology," Biophysical Journal 88(1) 623-638.

    Instead of measuring the peak intensity of the feature and calculating the
    total intensity (assuming a certain feature shape), the total intensity
    (=mass) is summed directly from the data. This quantity is more robust
    to noise and gives a better estimate of the static error.

    In addition, the sum of squared coordinates is calculated by taking the
    discrete sum instead of taking the continuous limit and integrating. This
    makes it easier to implement anisotropic and n-dimensional masks.
    """
    noise_size = validate_tuple(noise_size, len(radius))[::-1]
    radius = radius[::-1]
    N = N_binary_mask(radius, len(radius))
    coord_moments = root_sum_x_squared(radius, len(radius))
    # If this is just one frame black_level is a scalar.
    mass_corr = mass - N * black_level
    N_S = noise / mass_corr
    if (radius[1:] == radius[:-1]) and (noise_size[1:] == noise_size[:-1]):
        ep = N_S * noise_size[0] * coord_moments[0]
    else:
        coord_moments = np.array(noise_size) * np.array(coord_moments)
        ep = N_S[:, np.newaxis] * coord_moments[np.newaxis, :]
    return ep
