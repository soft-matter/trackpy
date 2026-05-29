import os
import logging

import numpy as np
from numpy.testing import assert_allclose

import trackpy as tp
from trackpy.artificial import (draw_features_brightfield,
                                gen_nonoverlapping_locations,
                                gen_connected_locations)
from trackpy.tests.common import sort_positions, StrictTestCase
from trackpy.feature import locate
from trackpy.locate_functions.brightfield_ring import locate_brightfield_ring
from trackpy.refine.brightfield_ring import (_min_edge, _fit_circle)

path, _ = os.path.split(os.path.abspath(__file__))

# we need to use a low value for min_percentile because the artificial
# edge is very sharp
MIN_PERC = 0.5

def draw_artificial_image(shape, pos, radius, noise_level, dip=False,
                          traditional=False, **kwargs):
    radius = tp.utils.validate_tuple(radius, len(shape))

    # tp.locate ignores a margin of size radius, take 1 px more to be safe
    diameter = tuple([(r * 2) + 1 for r in radius])
    size = [d / 2 for d in diameter]

    cols = ['x', 'y', 'z'][:len(shape)][::-1]

    image = draw_features_brightfield(shape, pos, size, noise_level, dip=dip)

    if not traditional:
        kwargs.update({'min_percentile': MIN_PERC})
        result = locate_brightfield_ring(image, diameter, **kwargs)
    else:
        result = locate(image, diameter, **kwargs)

    # For some reason, sorting the DataFrame gives wrong orders in some cases
    result = np.sort(result[cols].astype(float).values, axis=0)
    expected = np.sort(pos, axis=0)

    return result, expected

def artificial_image(shape, count, radius, noise_level, dip=False,
                     traditional=False, **kwargs):

    radius = tp.utils.validate_tuple(radius, len(shape))
    margin = tuple([r + 1 for r in radius])
    separation = tuple([2.5*r for r in radius])
    pos = gen_nonoverlapping_locations(shape, count, separation, margin)
    return draw_artificial_image(shape, pos, radius, noise_level, dip,
                                 traditional, **kwargs)

def artificial_cluster(shape, count, radius, noise_level, dip=False,
                       traditional=False, **kwargs):
    radius = tp.utils.validate_tuple(radius, len(shape))
    margin = tuple([r + 1 for r in radius])
    separation = tuple([1.4*r for r in radius])
    pos = gen_connected_locations(shape, count, separation, margin)
    return draw_artificial_image(shape, pos, radius, noise_level, dip,
                                 traditional, **kwargs)

def generate_random_circle(r, x, y, num_samples=500, noise=0):
    np.random.seed(1)
    theta = np.random.rand(num_samples) * (2 * np.pi)

    if noise > 0:
        mini = r-noise
        maxi = r+noise
        r_rand = np.random.rand(num_samples) * (maxi-mini) + mini
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
        self.n_feat_cluster = 3
        self.n_feat_dense = 30
        self.image_size = (250, 350)
        self.radius = 13
        self.cluster_sep = 1.0*self.radius

    def test_multiple_simple_sparse(self):
        actual, expected = artificial_image(self.image_size,
                                            self.n_feat_sparse,
                                            self.radius, noise_level=0)
        assert_allclose(actual, expected, atol=self.pixel_tolerance)

    def test_multiple_simple_sparse_no_multiprocessing(self):
        actual, expected = artificial_image(self.image_size,
                                            self.n_feat_sparse,
                                            self.radius, noise_level=0,
                                            processes=0)
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

    def test_cluster(self):
        actual, expected = artificial_cluster(self.image_size,
                                              self.n_feat_cluster,
                                              self.radius, noise_level=0,
                                              separation=self.cluster_sep)
        assert_allclose(actual, expected, atol=self.pixel_tolerance)

    def test_cluster_noisy(self):
        actual, expected = artificial_cluster(self.image_size,
                                              self.n_feat_cluster,
                                              self.radius, noise_level=10,
                                              separation=self.cluster_sep)
        assert_allclose(actual, expected, atol=self.pixel_tolerance)

    def test_cluster_more_noisy(self):
        actual, expected = artificial_cluster(self.image_size,
                                              self.n_feat_cluster,
                                              self.radius, noise_level=51,
                                              separation=self.cluster_sep)
        assert_allclose(actual, expected, atol=self.pixel_tolerance)

    def test_cluster_dip(self):
        actual, expected = artificial_cluster(self.image_size,
                                              self.n_feat_cluster,
                                              self.radius, noise_level=0,
                                              dip=True,
                                              separation=self.cluster_sep)
        assert_allclose(actual, expected, atol=self.pixel_tolerance)

    def test_cluster_noisy_dip(self):
        actual, expected = artificial_cluster(self.image_size,
                                              self.n_feat_cluster,
                                              self.radius, noise_level=10,
                                              dip=True,
                                              separation=self.cluster_sep)
        assert_allclose(actual, expected, atol=self.pixel_tolerance)

    def test_cluster_more_noisy_dip(self):
        actual, expected = artificial_cluster(self.image_size,
                                              self.n_feat_cluster,
                                              self.radius, noise_level=51,
                                              dip=True,
                                              separation=self.cluster_sep)
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
        image = np.zeros(self.image_size, dtype=float)

        ix = int(np.round(float(self.image_size[1])/2.0))
        image[:, :ix] += 230.0
        image[:, [ix, ix+1]] = 0.0
        image[:, ix+2:] += 100.0

        result = _min_edge(image, 0.45, 2, min_percentile=MIN_PERC)
        assert_allclose(result, float(ix)+0.5, atol=0.1)

    def test_min_edge_noisy(self):
        image = np.zeros(self.image_size, dtype=float)

        ix = int(np.round(float(self.image_size[1])/2.0))
        image[:, :ix] += np.random.uniform(150.0, 255.0, image[:, :ix].shape)
        image[:, [ix, ix+1]] += np.random.uniform(0.0, 50.0, image[:, [ix, ix+1]].shape)
        image[:, (ix+2):] += np.random.uniform(80.0, 110.0, image[:, (ix+2):].shape)

        result = _min_edge(image, 0.45, 2, min_percentile=MIN_PERC)
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
    import unittest
    unittest.main()
