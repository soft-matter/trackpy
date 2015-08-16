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
