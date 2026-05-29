import numpy as np
from scipy.ndimage import binary_dilation
from pandas import DataFrame

from .masks import binary_mask, x_squared_masks
from .utils import memo, validate_tuple


def measure_noise(image_bp, image_raw, radius):
    """Compute the mean and standard deviation of the dark pixels outside the
    signal. The bandpassed image is used to identify background regions. The
    raw image is used to characterize the background.
    See Biophysical journal 88(1) 623-638 Figure C.

    Parameters
    ----------
    image_bp : ndarray
        preprocessed (bandpassed) image
    image_raw : ndarray
        raw image
    radius : number or tuple of numbers
        feature radius used for centroid identification

    Returns
    -------
    background mean, background standard deviation
    """
    structure = binary_mask(radius, image_bp.ndim)
    background = ~binary_dilation(image_bp, structure=structure)
    n_background = background.sum()
    if n_background == 0:  # edge case of no background identified
        return np.nan, np.nan
    elif n_background == 1:  # edge case of not enough background identified
        return image_raw[background].mean(), np.nan
    else:
        return image_raw[background].mean(), image_raw[background].std()


@memo
def _root_sum_x_squared(radius, ndim):
    "Returns the root of the sum of all x^2 inside the mask for each dim."
    masks = x_squared_masks(radius, ndim)
    r2 = np.sum(masks, axis=tuple(range(1, ndim + 1)))  # each ax except first
    return np.sqrt(r2)


def _static_error(mass, noise, radius, noise_size):
    coord_moments = _root_sum_x_squared(radius, len(radius))
    N_S = noise / mass
    if np.all(radius[1:] == radius[:-1]) and \
       np.all(noise_size[1:] == noise_size[:-1]):
        ep = N_S * noise_size[0] * coord_moments[0]
    else:
        ep = N_S[:, np.newaxis] * \
             (np.array(noise_size) * np.array(coord_moments))[np.newaxis, :]
    return ep


def static_error(features, noise, diameter, noise_size=1, ndim=2):
    """Compute the uncertainty in particle position ("the static error").

    Parameters
    ----------
    features : DataFrame of features
        The feature dataframe should have a `mass` column that is already
        background corrected.
    noise : number or DataFrame having `noise` column, indexed on `frame`
        standard deviation of the noise
    diameter : number or tuple, feature diameter used to locate centroids
    noise_size : noise correlation length, may be tuple-valued
    ndim : number of image dimensions, default 2
        if diameter is tuple-valued then its length will override ndim

    Returns
    -------
    DataFrame of static error estimates, indexed like the features.
    When either radius or noise_size are anisotropic, the returned DataFrame
    contains one column for each dimension.

    Where uncertainty estimation fails, NaN is returned.

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
    if hasattr(diameter, '__iter__'):
        ndim = len(diameter)
    noise_size = validate_tuple(noise_size, ndim)[::-1]
    diameter = validate_tuple(diameter, ndim)[::-1]
    radius = tuple([d // 2 for d in diameter])

    if np.isscalar(noise):
        ep = _static_error(features['mass'], noise, radius, noise_size)
    else:
        assert 'noise' in noise
        temp = features.join(noise, on='frame')
        ep = _static_error(temp['mass'], temp['noise'], radius, noise_size)

    ep[ep < 0] = np.nan

    if ep.ndim == 1:
        ep.name = 'ep'
    elif ep.ndim == 2:
        if ndim < 4:
            coord_columns = ['ep_x', 'ep_y', 'ep_z'][:ndim]
        else:
            coord_columns = map(lambda i: 'ep_x' + str(i), range(ndim))
        ep = DataFrame(ep, columns=coord_columns, index=features.index)
    return ep
