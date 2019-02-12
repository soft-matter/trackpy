from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
from six.moves import range
import os
import warnings
import logging

import nose
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from numpy.testing import assert_allclose, assert_array_less
from pandas.util.testing import assert_produces_warning

import trackpy as tp
from trackpy.artificial import (draw_features_brightfield,
                                gen_nonoverlapping_locations)
from trackpy.utils import pandas_sort
from trackpy.tests.common import sort_positions, StrictTestCase
from trackpy.feature import locate
from trackpy.locate_functions.brightfield_ring import locate_brightfield_ring
from trackpy.refine.brightfield_ring import (_min_edge, _fit_circle)

path, _ = os.path.split(os.path.abspath(__file__))


def artificial_image(shape, count, radius, noise_level, dip=False, 
                     traditional=False, **kwargs):
    radius = tp.utils.validate_tuple(radius, len(shape))

    # tp.locate ignores a margin of size radius, take 1 px more to be safe
    margin = tuple([r + 1 for r in radius])
    diameter = tuple([(r * 2) + 1 for r in radius])
    size = [d / 2 for d in diameter]
    separation = tuple([d * 1.1 for d in diameter])

    cols = ['x', 'y', 'z'][:len(shape)][::-1]

    pos = gen_nonoverlapping_locations(shape, count, separation, margin)
    image = draw_features_brightfield(shape, pos, size, noise_level, dip=dip)

    if not traditional:
        result = locate_brightfield_ring(image, diameter, **kwargs)
    else:
        result = locate(image, diameter, **kwargs)

    # For some reason, sorting the DataFrame gives wrong orders in some cases
    result = np.sort(result[cols].astype(float).values, axis=0)
    expected = np.sort(pos, axis=0)

    return result, expected

def generate_random_circle(r, x, y, num_samples=500, noise=0):
    np.random.seed(1)
    theta = np.random.rand((num_samples)) * (2 * np.pi)

    if noise > 0:
        mini = r-noise
        maxi = r+noise
        r_rand = np.random.rand((num_samples)) * (maxi-mini) + mini
    else:
        r_rand = r

    xc = r_rand * np.cos(theta) + x
    yc = r_rand * np.sin(theta) + y

    return np.squeeze(np.dstack((xc, yc))).T

class TestLocateBrightfieldRing(StrictTestCase):
    def setUp(self):
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)

        self.pixel_tolerance = 0.9
        self.n_feat_sparse = 5
        self.n_feat_dense = 30
        self.image_size = (250, 350)
        self.radius = 11

    def test_multiple_simple_sparse(self):
        actual, expected = artificial_image(self.image_size,
                                            self.n_feat_sparse,
                                            self.radius, noise_level=0)
        assert_allclose(actual, expected, atol=self.pixel_tolerance)

    def test_multiple_simple_sparse_dip(self):
        actual, expected = artificial_image(self.image_size,
                                            self.n_feat_sparse,
                                            self.radius, noise_level=0,
                                            dip=True)
        assert_allclose(actual, expected, atol=self.pixel_tolerance)

    def test_multiple_noisy_sparse(self):
        actual, expected = artificial_image(self.image_size,
                                            self.n_feat_sparse,
                                            self.radius, noise_level=10)
        assert_allclose(actual, expected, atol=self.pixel_tolerance)

    def test_multiple_noisy_sparse_dip(self):
        actual, expected = artificial_image(self.image_size,
                                            self.n_feat_sparse,
                                            self.radius, noise_level=10,
                                            dip=True)
        assert_allclose(actual, expected, atol=self.pixel_tolerance)

    def test_multiple_more_noisy_sparse(self):
        actual, expected = artificial_image(self.image_size,
                                            self.n_feat_sparse,
                                            self.radius, noise_level=51)
        assert_allclose(actual, expected, atol=self.pixel_tolerance)

    def test_multiple_more_noisy_sparse_dip(self):
        actual, expected = artificial_image(self.image_size,
                                            self.n_feat_sparse,
                                            self.radius, noise_level=51,
                                            dip=True)
        assert_allclose(actual, expected, atol=self.pixel_tolerance)

    def test_multiple_simple_dense(self):
        actual, expected = artificial_image(self.image_size,
                                            self.n_feat_dense,
                                            self.radius, noise_level=0)
        assert_allclose(actual, expected, atol=self.pixel_tolerance)

    def test_multiple_simple_dense_dip(self):
        actual, expected = artificial_image(self.image_size,
                                            self.n_feat_dense,
                                            self.radius, noise_level=0,
                                            dip=True)
        assert_allclose(actual, expected, atol=self.pixel_tolerance)

    def test_multiple_noisy_dense(self):
        actual, expected = artificial_image(self.image_size,
                                            self.n_feat_dense,
                                            self.radius, noise_level=10)
        assert_allclose(actual, expected, atol=self.pixel_tolerance)

    def test_multiple_noisy_dense_dip(self):
        actual, expected = artificial_image(self.image_size,
                                            self.n_feat_dense,
                                            self.radius, noise_level=10,
                                            dip=True)
        assert_allclose(actual, expected, atol=self.pixel_tolerance)

    def test_multiple_more_noisy_dense(self):
        actual, expected = artificial_image(self.image_size,
                                            self.n_feat_dense,
                                            self.radius, noise_level=51)
        assert_allclose(actual, expected, atol=self.pixel_tolerance)

    def test_multiple_more_noisy_dense_dip(self):
        actual, expected = artificial_image(self.image_size,
                                            self.n_feat_dense,
                                            self.radius, noise_level=51,
                                            dip=True)
        assert_allclose(actual, expected, atol=self.pixel_tolerance)


    def test_default_locate_multiple_more_noisy_dense_dip(self):
        # This shows where the locate function fails with the same parameters,
        # for example
        actual, expected = artificial_image(self.image_size,
                                            self.n_feat_dense,
                                            self.radius, noise_level=51,
                                            dip=True, traditional=True)

        equal_shape = (actual.shape == expected.shape)

        if equal_shape:
            assert not np.allclose(actual, expected, atol=self.pixel_tolerance)
        else:
            assert not equal_shape

    def test_min_edge(self):
        image = np.ones(self.image_size, dtype=np.float)
        ix = int(np.round(float(self.image_size[1])/2.0))
        image[:, [ix, ix+1]] = 0.0

        result = _min_edge(image, 0.5)
        assert_allclose(result, float(ix)+0.5, atol=0.1)

    def test_min_edge_noisy(self):
        image = np.zeros(self.image_size, dtype=np.float)
        image += np.random.randint(1, 255, image.shape).astype(float)

        ix = int(np.round(float(self.image_size[1])/2.0))
        image[:, [ix, ix+1]] = 0.0

        result = _min_edge(image, 0.5)
        assert_allclose(result, float(ix)+0.5, atol=0.1)

    def test_fit_circle(self):
        x = 40.5
        y = 90.3
        circle_coords = generate_random_circle(self.radius, x, y, noise=0)

        r_result, (xc, yc) = _fit_circle(circle_coords)
        assert_allclose(r_result, self.radius, atol=0.9)
        assert_allclose(xc, x, atol=0.9)
        assert_allclose(yc, y, atol=0.9)

    def test_fit_circle_noisy(self):
        x = 40.5
        y = 90.3
        circle_coords = generate_random_circle(self.radius, x, y, 
                                               noise=0.2*self.radius)

        r_result, (xc, yc) = _fit_circle(circle_coords)
        assert_allclose(r_result, self.radius, atol=0.9)
        assert_allclose(xc, x, atol=0.9)
        assert_allclose(yc, y, atol=0.9)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
