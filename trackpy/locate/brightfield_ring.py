""" Detect particles in brightfield mode by tracking a ring of dark pixels
    around a bright interior part. """
from __future__ import division, print_function, absolute_import

import warnings
import numpy as np
from pandas import (DataFrame,)

from ..find import (grey_dilation, where_close)
from ..refine import (refine_brightfield_ring,)
from ..utils import (validate_tuple, default_pos_columns)
from ..preprocessing import (bandpass, convert_to_int)


def locate_brightfield_ring(raw_image, diameter, separation=None, noise_size=1,
                            smoothing_size=None, threshold=None,
                            previous_coords=None):
    """Locate particles imaged in brightfield mode of some approximate size in
    an image.

    Preprocess the image by performing a band pass and a threshold. Locate all
    peaks of brightness, then find the particle position by fitting the ring of
    dark pixels around the bright inner part of the particle.

    Parameters
    ----------
    raw_image : array
         any N-dimensional image
    diameter : odd integer or tuple of odd integers
        This may be a single number or a tuple giving the feature's
        extent in each dimension, useful when the dimensions do not have
        equal resolution (e.g. confocal microscopy). The tuple order is the
        same as the image shape, conventionally (z, y, x) or (y, x). The
        number(s) must be odd integers. When in doubt, round up.
    separation : float or tuple
        Minimum separtion between features.
        Default is diameter + 1. May be a tuple, see diameter for details.
    noise_size : float or tuple
        Width of Gaussian blurring kernel, in pixels
        Default is 1. May be a tuple, see diameter for details.
    smoothing_size : float or tuple
        The size of the sides of the square kernel used in boxcar (rolling
        average) smoothing, in pixels
        Default is diameter. May be a tuple, making the kernel rectangular.
    threshold : float
        Clip bandpass result below this value. Thresholding is done on the
        already background-subtracted image.
        By default, 1 for integer images and 1/255 for float images.
    previous_coords : DataFrame([x, y, size])
        Optional previous particle positions from the preceding frame to use as
        starting point for the refinement instead of the intensity peaks.

    Returns
    -------
    DataFrame([x, y, size])
        where size means the radius of the fitted circle of dark pixels around
        the bright interior of the particle.

    See Also
    --------
    refine_brightfield_ring : performs the refinement of the particle position

    Notes
    -----
    Locate works with a coordinate system that has its origin at the center of
    pixel (0, 0). In almost all cases this will be the topleft pixel: the
    y-axis is pointing downwards.

    This is an implementation of an algorithm described in [1]_

    References
    ----------
    .. [1] M. Rinaldin, R.W. Verweij, I. Chakraborty and D.J. Kraft, Soft
           Matter, 2019, 15, 1345-1360, http://dx.doi.org/10.1039/C8SM01661E

    """
    # Validate parameters and set defaults.
    raw_image = np.squeeze(raw_image)
    shape = raw_image.shape
    ndim = len(shape)

    diameter = validate_tuple(diameter, ndim)
    diameter = tuple([int(x) for x in diameter])
    if not np.all([x & 1 for x in diameter]):
        raise ValueError("Feature diameter must be an odd integer. Round up.")
    radius = tuple([x//2 for x in diameter])

    is_float_image = not np.issubdtype(raw_image.dtype, np.integer)

    if separation is None:
        separation = tuple([x + 1 for x in diameter])
    else:
        separation = validate_tuple(separation, ndim)

    if smoothing_size is None:
        smoothing_size = diameter
    else:
        smoothing_size = validate_tuple(smoothing_size, ndim)

    noise_size = validate_tuple(noise_size, ndim)

    # Check whether the image looks suspiciously like a color image.
    if 3 in shape or 4 in shape:
        dim = raw_image.ndim
        warnings.warn("I am interpreting the image as {0}-dimensional. "
                      "If it is actually a {1}-dimensional color image, "
                      "convert it to grayscale first.".format(dim, dim-1))

    if threshold is None:
        if is_float_image:
            threshold = 1/255.
        else:
            threshold = 1

    image = bandpass(raw_image, noise_size, smoothing_size, threshold)

    # For optimal performance, coerce the image dtype to integer.
    if is_float_image:  # For float images, assume bitdepth of 8.
        dtype = np.uint8
    else:   # For integer images, take original dtype
        dtype = raw_image.dtype

    # Normalize_to_int does nothing if image is already of integer type.
    _, image = convert_to_int(image, dtype)

    pos_columns = default_pos_columns(image.ndim)

    if previous_coords is None or len(previous_coords) == 0:
        # Find local maxima.
        # Define zone of exclusion at edges of image, avoiding
        #   - Features with incomplete image data ("radius")
        #   - Extended particles that cannot be explored during subpixel
        #       refinement ("separation")
        #   - Invalid output of the bandpass step ("smoothing_size")
        margin = tuple([max(rad, sep // 2 - 1, sm // 2) for (rad, sep, sm) in
                        zip(radius, separation, smoothing_size)])
        # Find features with minimum separation distance of `separation`. This
        # excludes detection of small features close to large, bright features
        # using the `maxsize` argument.
        coords = grey_dilation(image, separation, margin=margin, precise=False)

        coords_df = DataFrame(columns=pos_columns, data=coords)
    else:
        coords_df = previous_coords

    if len(coords_df) == 0:
        warnings.warn("No particles found in the image before refinement.")
        return coords_df

    coords_df = refine_brightfield_ring(image, radius, coords_df,
                                        pos_columns=pos_columns)

    # Flat peaks return multiple nearby maxima. Eliminate duplicates.
    if np.all(np.greater(separation, 0)):
        to_drop = where_close(coords_df[pos_columns], separation)
        coords_df.drop(to_drop, axis=0, inplace=True)
        coords_df.reset_index(drop=True, inplace=True)

    if len(coords_df) == 0:
        warnings.warn("No particles found in the image after refinement.")
        return coords_df

    # If this is a pims Frame object, it has a frame number.
    # Tag it on; this is helpful for parallelization.
    if hasattr(raw_image, 'frame_no') and raw_image.frame_no is not None:
        coords_df['frame'] = int(raw_image.frame_no)

    return coords_df

