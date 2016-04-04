from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
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
    return np.sum(binary_mask(radius,ndim))


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


def _slice(lower, upper, shape):
    ndim = len(shape)
    origin = [None] * ndim
    slices = [None] * ndim
    for i, sh, low, up in zip(range(ndim), shape, lower, upper):
        lower_bound_trunc = max(0, low)
        upper_bound_trunc = min(sh, up)
        slices[i] = slice(lower_bound_trunc, upper_bound_trunc)
        origin[i] = lower_bound_trunc
    return slices, origin


def _in_bounds(coords, shape, radius):
    ndim = len(shape)
    in_bounds = np.array([(coords[:, i] >= -r) & (coords[:, i] < sh + r)
                         for i, sh, r in zip(range(ndim), shape, radius)])
    return coords[np.all(in_bounds, axis=0)]


def slices_multiple(coords, shape, radius):
    """Creates the smallest box so that every coord in `coords` is in the box
    up to `radius` from the coordinate."""
    ndim = len(shape)
    radius = validate_tuple(radius, ndim)
    coords = np.atleast_2d(np.round(coords).astype(np.int))
    coords = _in_bounds(coords, shape, radius)

    if len(coords) == 0:
        return [slice(None, 0)] * ndim, None

    return _slice(coords.min(axis=0) - radius,
                  coords.max(axis=0) + radius + 1, shape)


def slice_image(coords, image, radius):
    """Creates the smallest box so that every coord in `coords` is in the box
    up to `radius` from the coordinate."""
    slices, origin = slices_multiple(coords, image.shape, radius)
    return image[slices], origin  # mask origin


def binary_mask_multiple(coords_rel, shape, radius, include_edge=True):
    """Creates multiple elliptical masks.

    Parameters
    ----------
    coords_rel : ndarray (N x 2 or N x 3)
        coordinates
    shape : tuple
        shape of the image
    radius : number or tuple of number
        size of the masks
    """
    ndim = len(shape)
    radius = validate_tuple(radius, ndim)
    coords_rel = np.atleast_2d(coords_rel)

    if include_edge:
        dist = [np.sum(((np.indices(shape).T - coord) / radius)**2, -1) <= 1
                for coord in coords_rel]
    else:
        dist = [np.sum(((np.indices(shape).T - coord) / radius)**2, -1) < 1
                for coord in coords_rel]
    return np.any(dist, axis=0).T


def mask_image(coords, image, radius, origin=None, invert=False):
    """Masks an image with elliptical masks with size `radius`. At every coord
    in `coords`, the mask is applied to the image. When invert=True, the coords
    instead of the background will be made 0.
    Optionally, specify the topleft coordinate (origin) of the image."""
    if origin is not None:
        coords_rel = coords - np.array(origin)[np.newaxis, :]
    else:
        coords_rel = coords

    mask_cluster = binary_mask_multiple(coords_rel, image.shape, radius,
                                        include_edge=(not invert))

    if invert:
        return image * ~mask_cluster
    else:
        return image * mask_cluster
