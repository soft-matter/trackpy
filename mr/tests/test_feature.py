from __future__ import division
import os
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import mr

import unittest
import nose
from numpy.testing import assert_almost_equal, assert_allclose
from numpy.testing.decorators import slow
from pandas.util.testing import (assert_series_equal, assert_frame_equal)

path, _ = os.path.split(os.path.abspath(__file__))

def draw_gaussian_spot(image, pos, r):
    assert image.shape[0] != image.shape[1], \
        "For stupid numpy broadcasting reasons, don't make the image square."
    x, y = np.meshgrid(*np.array(map(np.arange, image.shape)) - pos)
    max_value = np.iinfo(image.dtype).max
    spot = max_value*np.exp(-(x**2 + y**2)/r).T
    image += spot

def gen_random_locations(shape, count):
    np.random.seed(0)
    return np.array([map(np.random.randint, shape) for _ in xrange(count)])

def draw_spots(shape, locations, r):
    image = np.zeros(shape, dtype='uint8')
    for x in locations:
        draw_gaussian_spot(image, x, r)
    return image

def compare(shape, count, radius):
    pos = gen_random_locations(shape, count) 
    image = draw_spots(shape, pos, radius)
    f = mr.locate(image, 2*radius + 1)
    actual = f[['x', 'y']].sort(['x', 'y'])
    expected = DataFrame(pos, columns=['y', 'x'])[['x', 'y']].sort(['x', 'y']) 
    return actual, expected

class TestFeatureIdentification(unittest.TestCase):

    def setUp(self):
        self.features = pd.read_pickle(
            os.path.join(path, 'data', 'features_size9_masscut2000.df'))
        self.v = mr.ImageSequence(
            os.path.join(path, 'video', 'image_sequence'))

    def test_simple_sparse(self):
        actual, expected = compare((200, 300), 4, 3)
        assert_allclose(actual, expected, atol=0.5)

    def test_real_data(self):
        actual = mr.batch(self.v[:], 9, 2000)
        assert_allclose(actual, self.features, atol=0.5)
