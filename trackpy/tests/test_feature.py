from __future__ import division
import os
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import trackpy as tp

import unittest
import nose
from numpy.testing import assert_almost_equal, assert_allclose
from numpy.testing.decorators import slow
from pandas.util.testing import (assert_series_equal, assert_frame_equal)


path, _ = os.path.split(os.path.abspath(__file__))


def draw_gaussian_spot(image, pos, r, max_value=None, ecc=0):
    if image.shape[0] == image.shape[1]:
        raise ValueError("For stupid numpy broadcasting reasons, don't make" +
                         "the image square.")
    x, y = np.meshgrid(*np.array(map(np.arange, image.shape)) - pos)
    if max_value is None:
        max_value = np.iinfo(image.dtype).max - 3
    spot = max_value*np.exp(-((x/(1 - ecc))**2 + (y*(1 - ecc))**2)/(2*r**2)).T
    image += spot


def gen_random_locations(shape, count):
    np.random.seed(0)
    return np.array([map(np.random.randint, shape) for _ in xrange(count)])


def draw_spots(shape, locations, r, noise_level):
    np.random.seed(0)
    image = np.random.randint(0, 1 + noise_level, shape).astype('uint8')
    for x in locations:
        draw_gaussian_spot(image, x, r)
    return image


def compare(shape, count, radius, noise_level):
    pos = gen_random_locations(shape, count) 
    image = draw_spots(shape, pos, radius, noise_level)
    f = tp.locate(image, 2*radius + 1, minmass=1800)
    actual = f[['x', 'y']].sort(['x', 'y'])
    expected = DataFrame(pos, columns=['y', 'x'])[['x', 'y']].sort(['x', 'y']) 
    return actual, expected


class CommonFeatureIdentificationTests(object):

    def check_skip(self):
        pass

    def test_smoke_test(self):
        self.check_skip()
        # simple "smoke" test to see if numba explodes
        dummy_image = np.random.randint(0, 100, (300, 300)).astype(np.uint8)
        tp.locate(dummy_image, 5)

    def test_one_centered_gaussian(self):
        self.check_skip()
        L = 21
        dims = (L, L + 2)  # avoid square images in tests
        pos = np.array([7, 13])
        cols = ['x', 'y']
        expected = DataFrame(pos.reshape(1, -1), columns=cols)

        image = np.ones(dims, dtype='uint8')
        draw_gaussian_spot(image, pos[::-1], 4)
        actual = tp.locate(image, 9, 1, preprocess=False)[cols]
        assert_allclose(actual, expected, atol=0.1)

    def test_subpx_precision(self): 
        self.check_skip()
        L = 21
        dims = (L, L + 2)  # avoid square images in tests
        cols = ['x', 'y']
        PRECISION = 0.1

        # one bright pixel
        pos = np.array([7, 13])
        image = np.ones(dims, dtype='uint8')
        image[tuple(pos[::-1])] = 100
        actual = tp.locate(image, 3, 1, preprocess=False)[cols]
        expected = DataFrame(pos.reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=0.001)  # especially precise

        # two neighboring pixels of equal brightness
        pos1 = np.array([7, 13])
        pos2 = np.array([8, 13])
        pos = np.array([7.5, 13])  # center is between pixels
        image = np.ones(dims, dtype='uint8')
        image[tuple(pos1[::-1])] = 100
        image[tuple(pos2[::-1])] = 100
        actual = tp.locate(image, 5, 1, preprocess=False)[cols]
        expected = DataFrame(pos.reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        # two neighboring pixels of unequal brightness
        pos1 = np.array([7, 13])
        pos2 = np.array([8, 13])
        pos = np.array([7.25, 13])  # center is between pixels, biased left
        image = np.ones(dims, dtype='uint8')
        image[tuple(pos1[::-1])] = 100
        image[tuple(pos2[::-1])] = 50
        actual = tp.locate(image, 5, 1, preprocess=False)[cols]
        expected = DataFrame(pos.reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        pos = np.array([7.75, 13])  # center is between pixels, biased right 
        image = np.ones(dims, dtype='uint8')
        image[tuple(pos1[::-1])] = 50
        image[tuple(pos2[::-1])] = 100
        actual = tp.locate(image, 5, 1, preprocess=False)[cols]
        expected = DataFrame(pos.reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        pos1 = np.array([7, 12])
        pos2 = np.array([7, 13])
        pos = np.array([7, 12.25])  # center is between pixels, biased down
        image = np.ones(dims, dtype='uint8')
        image[tuple(pos1[::-1])] = 100
        image[tuple(pos2[::-1])] = 50
        actual = tp.locate(image, 5, 1, preprocess=False)[cols]
        expected = DataFrame(pos.reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        pos = np.array([7, 12.75])  # center is between pixels, biased up 
        image = np.ones(dims, dtype='uint8')
        image[tuple(pos1[::-1])] = 50
        image[tuple(pos2[::-1])] = 100
        actual = tp.locate(image, 5, 1, preprocess=False)[cols]
        expected = DataFrame(pos.reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        # four neighboring pixels of unequal brightness
        pos1 = np.array([7, 13])
        pos2 = np.array([8, 13])
        pos3 = np.array([7, 13])
        pos4 = np.array([8, 13])
        pos = np.array([7.25, 13])  # center is between pixels, biased left
        image = np.ones(dims, dtype='uint8')
        image[tuple(pos1[::-1])] = 100
        image[tuple(pos2[::-1])] = 50
        image[tuple(pos3[::-1])] = 100
        image[tuple(pos4[::-1])] = 50
        actual = tp.locate(image, 5, 1, preprocess=False)[cols]
        expected = DataFrame(pos.reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        pos = np.array([7.75, 13])  # center is between pixels, biased right 
        image = np.ones(dims, dtype='uint8')
        image[tuple(pos1[::-1])] = 50
        image[tuple(pos2[::-1])] = 100
        image[tuple(pos3[::-1])] = 50
        image[tuple(pos4[::-1])] = 100
        actual = tp.locate(image, 5, 1, preprocess=False)[cols]
        expected = DataFrame(pos.reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        pos1 = np.array([7, 12])
        pos2 = np.array([7, 13])
        pos3 = np.array([7, 12])
        pos4 = np.array([7, 13])
        pos = np.array([7, 12.25])  # center is between pixels, biased down
        image = np.ones(dims, dtype='uint8')
        image[tuple(pos1[::-1])] = 100
        image[tuple(pos2[::-1])] = 50
        actual = tp.locate(image, 5, 1, preprocess=False)[cols]
        expected = DataFrame(pos.reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        pos = np.array([7, 12.75])  # center is between pixels, biased up 
        image = np.ones(dims, dtype='uint8')
        image[tuple(pos1[::-1])] = 50
        image[tuple(pos2[::-1])] = 100
        actual = tp.locate(image, 5, 1, preprocess=False)[cols]
        expected = DataFrame(pos.reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

    def test_multiple_simple_sparse(self):
        self.check_skip()
        actual, expected = compare((200, 300), 4, 2, noise_level=0)
        assert_allclose(actual, expected, atol=0.5)

    def test_multiple_noisy_sparse(self):
        self.check_skip()
        actual, expected = compare((200, 300), 4, 2, noise_level=1)
        assert_allclose(actual, expected, atol=0.5)

    def test_multiple_more_noisy_sparse(self):
        self.check_skip()
        actual, expected = compare((200, 300), 4, 2, noise_level=2)
        assert_allclose(actual, expected, atol=0.5)

    def test_topn(self):
        self.check_skip()
        L = 21
        dims = (L, L + 2)  # avoid square images in tests
        cols = ['x', 'y']
        PRECISION = 0.1

        # top 2
        pos1 = np.array([7, 7])
        pos2 = np.array([14, 14])
        pos3 = np.array([7, 14])
        image = np.ones(dims, dtype='uint8')
        image[tuple(pos1[::-1])] = 100
        image[tuple(pos2[::-1])] = 80
        image[tuple(pos3[::-1])] = 90
        actual = tp.locate(image, 5, 1, topn=2, preprocess=False)[cols]
        actual = actual.sort(['x', 'y'])  # sort for reliable comparison
        expected = DataFrame([[7, 7], [7, 14]], columns=cols).sort(['x', 'y'])
        assert_allclose(actual, expected, atol=PRECISION)

        # top 1
        actual = tp.locate(image, 5, 1, topn=1, preprocess=False)[cols]
        actual = actual.sort(['x', 'y'])  # sort for reliable comparison
        expected = DataFrame([[7, 7]], columns=cols).sort(['x', 'y'])
        assert_allclose(actual, expected, atol=PRECISION)

    def test_rg(self):
        # For Gaussians with radii 2, 3, 5, and 7 px, with proportionately
        # chosen feature (mask) sizes, the 'size' comes out to be within 10%
        # of the true Gaussian width.

        # The IDL code has mistake in this area, documented here:
        # http://www.physics.emory.edu/~weeks/idl/radius.html

        self.check_skip()
        L = 101 
        dims = (L, L + 2)  # avoid square images in tests
        pos = np.array([50, 55])
        cols = ['x', 'y']

        SIZE = 2
        image = np.ones(dims, dtype='uint8')
        draw_gaussian_spot(image, pos[::-1], SIZE)
        actual = tp.locate(image, 7, 1, preprocess=False)['size']
        expected = SIZE
        assert_allclose(actual, expected, rtol=0.1)

        SIZE = 3
        image = np.ones(dims, dtype='uint8')
        draw_gaussian_spot(image, pos[::-1], SIZE)
        actual = tp.locate(image, 11, 1, preprocess=False)['size']
        expected = SIZE
        assert_allclose(actual, expected, rtol=0.1)

        SIZE = 5
        image = np.ones(dims, dtype='uint8')
        draw_gaussian_spot(image, pos[::-1], SIZE)
        actual = tp.locate(image, 17, 1, preprocess=False)['size']
        expected = SIZE
        assert_allclose(actual, expected, rtol=0.1)
        
        SIZE = 7
        image = np.ones(dims, dtype='uint8')
        draw_gaussian_spot(image, pos[::-1], SIZE)
        actual = tp.locate(image, 23, 1, preprocess=False)['size']
        expected = SIZE
        assert_allclose(actual, expected, rtol=0.1)
        
    def test_eccentricity(self):
        # Eccentricity (elongation) is measured with good accuracy and
        # ~0.02 precision, as long as the mask is large enough to cover
        # the whole object.
        self.check_skip()
        L = 101 
        dims = (L, L + 2)  # avoid square images in tests
        pos = np.array([50, 55])
        cols = ['x', 'y']

        ECC = 0
        image = np.ones(dims, dtype='uint8')
        draw_gaussian_spot(image, pos[::-1], 4, ecc=ECC)
        actual = tp.locate(image, 21, 1, preprocess=False)['ecc']
        expected = ECC
        assert_allclose(actual, expected, atol=0.02)

        ECC = 0.2
        image = np.ones(dims, dtype='uint8')
        draw_gaussian_spot(image, pos[::-1], 4, ecc=ECC)
        actual = tp.locate(image, 21, 1, preprocess=False)['ecc']
        expected = ECC
        assert_allclose(actual, expected, atol=0.02)

        ECC = 0.5
        image = np.ones(dims, dtype='uint8')
        draw_gaussian_spot(image, pos[::-1], 4, ecc=ECC)
        actual = tp.locate(image, 21, 1, preprocess=False)['ecc']
        expected = ECC
        assert_allclose(actual, expected, atol=0.02)


    def test_whole_pixel_shifts(self):
        self.check_skip()
        L = 21
        dims = (L, L + 2)  # avoid square images in tests
        pos = np.array([7, 13])
        cols = ['x', 'y']
        expected = np.array([pos])

        image = np.ones(dims, dtype='uint8')
        draw_gaussian_spot(image, pos[::-1], 2)

        guess = np.array([[6, 13]])
        actual = tp.feature.refine(image, image, 6, guess, characterize=False)
        assert_allclose(actual, expected, atol=0.1)

        guess = np.array([[7, 12]])
        actual = tp.feature.refine(image, image, 6, guess, characterize=False)
        assert_allclose(actual, expected, atol=0.1)

        guess = np.array([[7, 14]])
        actual = tp.feature.refine(image, image, 6, guess, characterize=False)
        assert_allclose(actual, expected, atol=0.1)

        guess = np.array([[6, 13]])
        actual = tp.feature.refine(image, image, 6, guess, characterize=False)
        assert_allclose(actual, expected, atol=0.1)

        guess = np.array([[6, 12]])
        actual = tp.feature.refine(image, image, 6, guess, characterize=False)
        assert_allclose(actual, expected, atol=0.1)


class TestFeatureIdentificationWithVanillaNumpy(
    CommonFeatureIdentificationTests, unittest.TestCase):

    def setUp(self):
        import trackpy as tp
        tp.disable_numba()


class TestFeatureIdentificationWithNumba(
    CommonFeatureIdentificationTests, unittest.TestCase):

    def setUp(self):
        import trackpy as tp
        try:
            import numba
        except ImportError:
            pass
            # Test will be skipped -- see check_skip()
        else:
            tp.enable_numba()  # enabled by default as of this writing

    def check_skip(self):
        try:
            import numba
        except ImportError:
            raise nose.SkipTest("Numba not installed. Skipping.")


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
