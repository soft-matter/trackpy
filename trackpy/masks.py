import numpy as np
from .utils import memo, validate_tuple

__all__ = ['binary_mask', 'r_squared_mask', 'cosmask', 'sinmask',
           'theta_mask']


@memo
def binary_mask(radius, ndim):
    "Elliptical mask in a rectangular array"
    radius = validate_tuple(radius, ndim)
    points = [np.arange(-rad, rad + 1) for rad in radius]
    if len(radius) > 1:
        coords = np.array(np.meshgrid(*points, indexing="ij"))
    else:
        coords = np.array([points[0]])
    r = [(coord/rad)**2 for (coord, rad) in zip(coords, radius)]
    return sum(r) <= 1


@memo
def N_binary_mask(radius, ndim):
    return np.sum(binary_mask(radius, ndim))


@memo
def r_squared_mask(radius, ndim):
    "Mask with values r^2 inside radius and 0 outside"
    radius = validate_tuple(radius, ndim)
    points = [np.arange(-rad, rad + 1) for rad in radius]
    if len(radius) > 1:
        coords = np.array(np.meshgrid(*points, indexing="ij"))
    else:
        coords = np.array([points[0]])
    r = [(coord/rad)**2 for (coord, rad) in zip(coords, radius)]
    r2 = np.sum(coords**2, 0).astype(int)
    r2[sum(r) > 1] = 0
    return r2
    

@memo
def x_squared_masks(radius, ndim):
    "Returns ndim masks with values x^2 inside radius and 0 outside"
    radius = validate_tuple(radius, ndim)
    points = [np.arange(-rad, rad + 1) for rad in radius]
    if len(radius) > 1:
        coords = np.array(np.meshgrid(*points, indexing="ij"))
    else:
        coords = np.array([points[0]])
    r = [(coord/rad)**2 for (coord, rad) in zip(coords, radius)]
    masks = np.asarray(coords**2, dtype=int)
    masks[:, sum(r) > 1] = 0
    return masks


@memo
def theta_mask(radius):
    """Mask of values giving angular position relative to center. The angle is
    defined according to ISO standards in which the angle is measured counter-
    clockwise from the x axis, measured in a normal coordinate system with y-
    axis pointing up and x axis pointing right.

    In other words: for increasing angle, the coordinate moves counterclockwise
    around the feature center starting on the right side.

    However, in most images, the y-axis will point down so that the coordinate
    will appear to move clockwise around the feature center.
    """
    # 2D only
    radius = validate_tuple(radius, 2)
    tan_of_coord = lambda y, x: np.arctan2(y - radius[0], x - radius[1])
    return np.fromfunction(tan_of_coord, [r * 2 + 1 for r in radius])


@memo
def sinmask(radius):
    "Sin of theta_mask"
    return np.sin(2*theta_mask(radius))


@memo
def cosmask(radius):
    "Sin of theta_mask"
    return np.cos(2*theta_mask(radius))


@memo
def gaussian_kernel(sigma, truncate=4.0):
    "1D discretized gaussian"
    lw = int(truncate * sigma + 0.5)
    x = np.arange(-lw, lw+1)
    result = np.exp(x**2/(-2*sigma**2))
    return result / np.sum(result)


def get_slice(coords, shape, radius):
    """Returns the slice and origin that belong to ``slice_image``"""
    # interpret parameters
    ndim = len(shape)
    radius = validate_tuple(radius, ndim)
    coords = np.atleast_2d(np.round(coords).astype(int))
    # drop features that have no pixels inside the image
    in_bounds = np.array([(coords[:, i] >= -r) & (coords[:, i] < sh + r)
                         for i, sh, r in zip(range(ndim), shape, radius)])
    coords = coords[np.all(in_bounds, axis=0)]
    # return if no coordinates are left
    if len(coords) == 0:
        return tuple([slice(None, 0)] * ndim), None
    # calculate the box
    lower = coords.min(axis=0) - radius
    upper = coords.max(axis=0) + radius + 1
    # calculate the slices
    origin = [None] * ndim
    slices = [None] * ndim
    for i, sh, low, up in zip(range(ndim), shape, lower, upper):
        lower_bound_trunc = max(0, low)
        upper_bound_trunc = min(sh, up)
        slices[i] = slice(int(round(lower_bound_trunc)),
                          int(round(upper_bound_trunc)))
        origin[i] = lower_bound_trunc
    return tuple(slices), origin


def slice_image(pos, image, radius):
    """ Slice a box around a group of features from an image.

    The box is the smallest box that contains all coordinates up to `radius`
    from any coordinate.

    Parameters
    ----------
    image : ndarray
        The image that will be sliced
    pos : iterable
        An iterable (e.g. list or ndarray) that contains the feature positions
    radius : number or tuple of numbers
        Defines the size of the slice. Every pixel that has a distance lower or
        equal to `radius` to a feature position is included.

    Returns
    -------
    tuple of:
    - the sliced image
    - the coordinate of the slice origin (top-left pixel)
    """
    slices, origin = get_slice(pos, image.shape,  radius)
    return image[slices], origin


def get_mask(pos, shape, radius, include_edge=True, return_masks=False):
    """ Create a binary mask that masks pixels farther than radius to all
    given feature positions.

    Optionally returns the masks that recover the individual feature pixels from
    a masked image, as follows: ``image[mask][masks_single[i]]``

    Parameters
    ----------
    pos : ndarray (N x 2 or N x 3)
        Feature positions
    shape : tuple
        The shape of the image
    radius : number or tuple
        Radius of the individual feature masks
    include_edge : boolean, optional
        Determine whether pixels at exactly one radius from a position are
        included. Default True.
    return_masks : boolean, optional
        Also return masks that recover the single features from a masked image.
        Default False.

    Returns
    -------
    ndarray containing a binary mask
    if return_masks==True, returns a tuple of [masks, masks_singles]
    """
    ndim = len(shape)
    radius = validate_tuple(radius, ndim)
    pos = np.atleast_2d(pos)

    if include_edge:
        in_mask = [np.sum(((np.indices(shape).T - p) / radius)**2, -1) <= 1
                   for p in pos]
    else:
        in_mask = [np.sum(((np.indices(shape).T - p) / radius)**2, -1) < 1
                   for p in pos]
    mask_total = np.any(in_mask, axis=0).T
    if return_masks:
        masks_single = np.empty((len(pos), mask_total.sum()), dtype=bool)
        for i, _in_mask in enumerate(in_mask):
            masks_single[i] = _in_mask.T[mask_total]
        return mask_total, masks_single
    else:
        return mask_total


def mask_image(pos, image, radius, origin=None, invert=False,
               include_edge=None):
    """ Masks an image so that pixels farther than radius to all given feature
    positions become 0.

    Parameters
    ----------
    pos : ndarray
        Feature positions (N x 2 or N x 3)
    image : ndarray
    radius : number or tuple
        Radius of the individual feature masks
    origin : tuple, optional
        The topleft coordinate (origin) of the image.
    invert : boolean, optional
        If invert==True, the features instead of the background will become 0.
    include_edge : boolean, optional
        Determine whether pixels at exactly one radius from a position are
        included in the feature mask.
        Defaults to True if invert==False, and to False if invert==True.
    """
    if origin is not None:
        pos = np.atleast_2d(pos) - np.array(origin)[np.newaxis, :]

    if include_edge is None:
        include_edge = not invert

    mask_cluster = get_mask(pos, image.shape, radius, include_edge=include_edge)

    if invert:
        mask_cluster = ~mask_cluster

    return image * mask_cluster.astype(np.uint8)
