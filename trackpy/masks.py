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
    if type(radius) == int: radius = (radius,) * ndim
    points = [range(-rad, rad + 1) for rad in radius]
    if len(radius) > 1:
        coords = np.array(np.meshgrid(*points, indexing="ij"))
    else:
        coords = np.array([points[0]])
    r = [(coord/rad)**2 for (coord,rad) in zip(coords,radius)]
    return sum(r) <= 1


@memo
def r_squared_mask(radius, ndim):
    "Mask with values r^2 inside radius and 0 outside"
    if type(radius) == int: radius = (radius,) * ndim
    points = [range(-rad, rad + 1) for rad in radius]
    if len(radius) > 1:
        coords = np.array(np.meshgrid(*points, indexing="ij"))
    else:
        coords = np.array([points[0]])
    r = [(coord/rad)**2 for (coord,rad) in zip(coords,radius)]
    r2 = np.sum(coords**2, 0).astype(int)
    r2[sum(r) > 1] = 0
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
