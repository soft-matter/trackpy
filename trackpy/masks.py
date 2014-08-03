from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

import numpy as np

from .utils import memo

@memo
def binary_mask(radius, ndim, separation=None):
    "circular mask in a square array"
    points = np.arange(-radius, radius + 1)
    if ndim > 1:
        coords = np.array(np.meshgrid(*([points]*ndim)))
    else:
        coords = points.reshape(1, -1)
    r = np.sqrt(np.sum(coords**2, 0))
    return r <= radius


@memo
def r_squared_mask(radius, ndim):
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
    # 2D only
    tan_of_coord = lambda y, x: np.arctan2(radius - y, x - radius)
    diameter = 2*radius + 1
    return np.fromfunction(tan_of_coord, (diameter, diameter))


@memo
def sinmask(radius):
    return np.sin(2*theta_mask(radius))


@memo
def cosmask(radius):
    return np.cos(2*theta_mask(radius))
