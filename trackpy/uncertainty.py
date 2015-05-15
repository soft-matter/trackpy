from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np
from scipy.ndimage import morphology

from .preprocessing import bandpass
from .masks import binary_mask, N_binary_mask, root_sum_x_squared
from .utils import validate_tuple


def roi(image, diameter, threshold=None, image_bandpassed=None):
    """Return a mask selecting the neighborhoods of bright regions.
    See Biophysical journal 88(1) 623-638 Figure C.

    Parameters
    ----------
    image : ndarray
    diameter : feature size used for centroid identification
    threshold : number, optional
    image_bandpassed : ndarray, optional

    Returns
    -------
    boolean ndarray, True around bright regions
    """
    diameter = validate_tuple(diameter, image.ndim)
    if image_bandpassed is None:
        image_bandpassed = bandpass(image, 1, tuple([d + 1 for d in diameter]),
                                    threshold)
    structure = binary_mask(tuple([int(d)//2 for d in diameter]), image.ndim)
    signal_mask = morphology.binary_dilation(image_bandpassed,
                                             structure=structure)
    return signal_mask


def measure_noise(image, diameter, threshold, image_bandpassed=None):
    "Compute the standard deviation of the dark pixels outside the signal."
    signal_mask = roi(image, diameter, threshold, image_bandpassed)
    return image[~signal_mask].mean(), image[~signal_mask].std()


def static_error(mass, noise, radius, ndim=2, noise_size=1):
    """Compute the uncertainty in particle position ("the static error").

    Parameters
    ----------
    mass : ndarray of feature masses, already background corrected
    noise : number, standard deviation of the noise
    radius : number or tuple, feature radius used to locate centroids
    ndim : number of image dimensions, default 2
    noise_size : noise correlation length, may be tuple-valued

    Returns
    -------
    1D or 2D array of static error estimates, indexed like the trajectories.
    When either radius or noise_size are anisotropic, a the returned array has
    ndim columns, one for each dimension. The order of these columns are equal
    to the order dimensions in radius and noise_size.

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
    makes it possible to generalize this analysis to anisotropic masks.
    """
    noise_size = validate_tuple(noise_size, ndim)
    radius = validate_tuple(radius, ndim)
    coord_moments = root_sum_x_squared(radius, ndim)
    isotropic = (np.all(radius[1:] == radius[:-1]) and
                 np.all(noise_size[1:] == noise_size[:-1]))
    if isotropic:
        ep = noise * noise_size[0] * coord_moments[0] / mass
    else:
        noise_moments = noise * np.array(noise_size) * np.array(coord_moments)
        ep = noise_moments[np.newaxis, :] / mass[:, np.newaxis]
    return ep
