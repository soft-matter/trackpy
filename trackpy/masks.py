from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

import numpy as np

from .utils import memo

__all__ = ['binary_mask', 'r_squared_mask', 'cosmask', 'sinmask',
           'theta_mask']


@memo
def binary_mask(radius, ndim):
    "Elliptical mask in a rectangular array"
    points = np.arange(-radius, radius + 1)
    if ndim > 1:
        coords = np.array(np.meshgrid(*([points]*ndim)))
    else:
        coords = points.reshape(1, -1)
    r = np.sqrt(np.sum(coords**2, 0))
    return r <= radius


@memo
def r_squared_mask(radius, ndim):
    "Mask with values r^2 inside radius and 0 outside"
    points = np.arange(-radius, radius + 1)
    if ndim > 1:
        coords = np.array(np.meshgrid(*([points]*ndim)))
    else:
        coords = points.reshape(1, -1)
    r2 = np.sum(coords**2, 0).astype(int)
    r2[r2 > radius**2] = 0
    return r2


@memo
def theta_mask(radius):
    "Mask of values giving angular position relative to center"
    # 2D only
    tan_of_coord = lambda y, x: np.arctan2(radius - y, x - radius)
    diameter = 2*radius + 1
    return np.fromfunction(tan_of_coord, (diameter, diameter))


@memo
def sinmask(radius):
    "Sin of theta_mask"
    return np.sin(2*theta_mask(radius))


@memo
def cosmask(radius):
    "Sin of theta_mask"
    return np.cos(2*theta_mask(radius))
