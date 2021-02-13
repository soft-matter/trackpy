"""
Detect particles in brightfield mode by tracking a ring of dark pixels around a
bright interior part. Based on https://github.com/caspervdw/circletracking
"""

import warnings
import numpy as np
from pandas import (DataFrame, concat)

from ..find import (grey_dilation, where_close)
from ..refine import (refine_brightfield_ring,)
from ..utils import (validate_tuple, default_pos_columns, get_pool)
from ..preprocessing import convert_to_int
from ..feature import locate


def locate_brightfield_ring(raw_image, diameter, separation=None,
                            previous_coords=None, processes='auto', **kwargs):
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
    previous_coords : DataFrame([x, y, r])
        Optional previous particle positions from the preceding frame to use as
        starting point for the refinement instead of the intensity peaks.
    processes : integer or "auto"
        The number of processes to use in parallel. If <= 1, multiprocessing is
        disabled. If "auto", the number returned by `os.cpu_count()`` is used.
    kwargs:
        Passed to the refine function.

    Returns
    -------
    DataFrame([x, y, r])
        where r means the radius of the fitted circle of dark pixels around
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
    diameter = tuple([float(x) for x in diameter])
    radius = tuple([x/2.0 for x in diameter])

    is_float_image = not np.issubdtype(raw_image.dtype, np.integer)

    if separation is None:
        separation = tuple([x for x in diameter])
    else:
        separation = validate_tuple(separation, ndim)

    # Check whether the image looks suspiciously like a color image.
    if 3 in shape or 4 in shape:
        dim = raw_image.ndim
        warnings.warn("I am interpreting the image as {}-dimensional. "
                      "If it is actually a {}-dimensional color image, "
                      "convert it to grayscale first.".format(dim, dim-1))

    image = raw_image

    # For optimal performance, coerce the image dtype to integer.
    if is_float_image:  # For float images, assume bitdepth of 8.
        dtype = np.uint8
    else:   # For integer images, take original dtype
        dtype = raw_image.dtype

    # Normalize_to_int does nothing if image is already of integer type.
    _, image = convert_to_int(image, dtype)

    pos_columns = default_pos_columns(image.ndim)

    has_user_input = False
    if previous_coords is None or len(previous_coords) == 0:
        coords_df = locate(raw_image, diameter, separation=separation,
                           characterize=False)
        coords_df = coords_df[pos_columns]
    else:
        coords_df = previous_coords
        has_user_input = True

    if len(coords_df) == 0:
        warnings.warn("No particles found in the image before refinement.")
        return coords_df

    pool, map_func = get_pool(processes)
    refined = []

    try:
        for result in map_func(_get_refined_coords, [(coords, pos_columns, image, radius, kwargs, has_user_input) for _, coords in coords_df.iterrows()]):
            if result is None:
                continue
            refined.append(result)
    finally:
        if pool:
            # Ensure correct termination of Pool
            pool.terminate()

    columns = np.unique(np.concatenate((pos_columns, ['r'], coords_df.columns)))
    if len(refined) == 0:
        warnings.warn("No particles found in the image after refinement.")
        return DataFrame(columns=columns)

    refined = DataFrame.from_dict(refined, orient='columns')
    refined.reset_index(drop=True, inplace=True)

    # Flat peaks return multiple nearby maxima. Eliminate duplicates.
    if np.all(np.greater(separation, 0)):
        to_drop = where_close(refined[pos_columns], separation)
        refined.drop(to_drop, axis=0, inplace=True)
        refined.reset_index(drop=True, inplace=True)

    # If this is a pims Frame object, it has a frame number.
    # Tag it on; this is helpful for parallelization.
    if hasattr(raw_image, 'frame_no') and raw_image.frame_no is not None:
        refined['frame'] = int(raw_image.frame_no)

    return refined

def _get_refined_coords(args):
    coords, pos_columns, image, radius, kwargs, has_user_input = args
    positions = coords[pos_columns]
    result = refine_brightfield_ring(image, radius, positions,
                                     pos_columns=pos_columns, **kwargs)
    if result is None:
        if has_user_input:
            warnings.warn(("Lost particle {:d} (x={:.0f}, y={:.0f})" +
                           " after refinement.").format(int(coords['particle']), coords['x'],
                                                        coords['y']))
        return None

    # Make a copy of old coords and overwrite with result
    # In this way any extra columns from previous_coords are preserved
    new_coords = coords.copy()
    for column in result.index.tolist():
        # make a new column if necessary, otherwise overwrite
        new_coords[column] = result.get(column)

    return new_coords

