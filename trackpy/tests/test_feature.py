from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
from six.moves import range
import os
import unittest
import warnings

import nose
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from numpy.testing import assert_almost_equal, assert_allclose
from numpy.testing.decorators import slow
from pandas.util.testing import (assert_series_equal, assert_frame_equal,
                                 assert_produces_warning)
from scipy.spatial import cKDTree

import trackpy as tp
from trackpy.try_numba import NUMBA_AVAILABLE
from trackpy.utils import validate_tuple


path, _ = os.path.split(os.path.abspath(__file__))


def feat_gauss(r, rg=0.333):
    """ Gaussian at r = 0 with max value of 1. Its radius of gyration is
    given by rg. """
    return np.exp((r/rg)**2 * r.ndim/-2)


def feat_gauss_edge(r, value_at_edge=0.1):
    """ Gaussian at r = 0 with max value of 1. Its value at r = 1 is given by
    value_at_edge. """
    return np.exp(np.log(value_at_edge)*r**2)


def feat_ring(r, r_at_max, value_at_edge=0.1):
    """ Ring feature with a gaussian profile, centered at r_at_max. Its value 
    at r = 1 is given by value_at_edge."""
    return np.exp(np.log(value_at_edge)*((r - r_at_max) / (1 - r_at_max))**2)


def feat_hat(r, disc_size, value_at_edge=0.1):
    """ Solid disc of size disc_size, with Gaussian smoothed borders. """
    mask = r > disc_size
    spot = (~mask).astype(r.dtype)
    spot[mask] = feat_ring(r[mask], disc_size, value_at_edge)
    spot[~mask] = 1
    return spot


def feat_step(r):
    """ Solid disc. """
    return r <= 1


def draw_feature(image, position, diameter, max_value=None,
                 feat_func=feat_gauss, ecc=None, **kwargs):
    """ Draws a radial symmetric feature and adds it to the image at given
    position.

    Parameters
    ----------
    image : ndarray
        image to draw features on
    position : iterable
        coordinates of feature position
    diameter : number
        defines the box that will be drawn on
    max_value : number
        maximum feature value. should be much less than the max value of the
        image dtype, to avoid pixel wrapping at overlapping features
    feat_func : function. Default: feat_gauss
        function f(r) that takes an ndarray of radius values
        and returns intensity values <= 1
    ecc : positive number, optional
        eccentricity of feature, defined only in 2D. Identical to setting
        diameter to (diameter / (1 - ecc), diameter * (1 - ecc))
    kwargs : keyword arguments are passed to feat_func
    """
    if len(position) != image.ndim:
        raise ValueError("Number of position coordinates should match image"
                         " dimensionality.")
    diameter = validate_tuple(diameter, image.ndim)
    if ecc is not None:
        if len(diameter) != 2:
            raise ValueError("Eccentricity is only defined in 2 dimensions")
        if diameter[0] != diameter[1]:
            raise ValueError("Diameter is already anisotropic; eccentricity is"
                             " not defined.")
        diameter = (diameter[0] / (1 - ecc), diameter[1] * (1 - ecc))
    radius = tuple([(d - 1) / 2 for d in diameter])
    if max_value is None:
        max_value = np.iinfo(image.dtype).max - 3
    rect = []
    vectors = []
    for (c, r, lim) in zip(position, radius, image.shape):
        if (c >= lim) or (c < 0):
            raise ValueError("Position outside of image.")
        lower_bound = max(int(np.floor(c - r)), 0)
        upper_bound = min(int(np.ceil(c + r + 1)), lim)
        rect.append(slice(lower_bound, upper_bound))
        vectors.append(np.arange(lower_bound - c, upper_bound - c) / r)
    coords = np.meshgrid(*vectors, indexing='ij', sparse=True)
    r = np.sqrt(np.sum(np.array(coords)**2, axis=0))
    spot = max_value * feat_func(r, **kwargs)
    image[rect] += spot.astype(image.dtype)


def draw_point(image, pos, value):
    image[tuple(pos)] = value


def gen_random_locations(shape, count, margin=0):
    margin = validate_tuple(margin, len(shape))
    np.random.seed(0)
    pos = [np.random.randint(round(m), round(s - m), count)
           for (s, m) in zip(shape, margin)]
    return np.array(pos).T


def eliminate_overlapping_locations(f, separation):
    separation = validate_tuple(separation, f.shape[1])
    assert np.greater(separation, 0).all()
    # Rescale positions, so that pairs are identified below a distance of 1.
    f = f / separation
    while True:
        duplicates = cKDTree(f, 30).query_pairs(1)
        if len(duplicates) == 0:
            break
        to_drop = []
        for pair in duplicates:
            to_drop.append(pair[1])
        f = np.delete(f, to_drop, 0)
    return f * separation


def gen_nonoverlapping_locations(shape, count, separation, margin=0):
    positions = gen_random_locations(shape, count, margin)
    return eliminate_overlapping_locations(positions, separation)


def draw_spots(shape, positions, diameter, noise_level=0, bitdepth=8,
               feat_func=feat_gauss, **kwargs):
    if bitdepth <= 8:
        dtype = np.uint8
        internaldtype = np.uint16
    elif bitdepth <= 16:
        dtype = np.uint16
        internaldtype = np.uint32
    elif bitdepth <= 32:
        dtype = np.uint32
        internaldtype = np.uint64
    else:
        raise ValueError('Bitdepth should be <= 32')
    np.random.seed(0)
    image = np.random.randint(0, noise_level + 1, shape).astype(internaldtype)
    for pos in positions:
        draw_feature(image, pos, diameter, max_value=2**bitdepth - 1,
                     feat_func=feat_func, **kwargs)
    return image.clip(0, 2**bitdepth - 1).astype(dtype)


def compare(shape, count, radius, noise_level, engine):
    radius = tp.utils.validate_tuple(radius, len(shape))
    # tp.locate ignores a margin of size radius, take 1 px more to be safe
    margin = tuple([r + 1 for r in radius])
    diameter = tuple([(r + 1) * 2 for r in radius])
    cols = ['x', 'y', 'z'][:len(shape)][::-1]
    pos = gen_nonoverlapping_locations(shape, count, diameter, margin)
    image = draw_spots(shape, pos, diameter, noise_level)
    f = tp.locate(image, [2*r + 1 for r in radius],
                  engine=engine)
    actual = f[cols].sort(cols)
    expected = DataFrame(pos, columns=cols).sort(cols)
    return actual, expected


class CommonFeatureIdentificationTests(object):

    def check_skip(self):
        pass

    def skip_numba(self):
        pass

    def test_smoke_datatypes(self):
        self.check_skip()
        SHAPE = (300, 300)
        # simple "smoke" test to see if numba explodes
        dummy_image = np.random.randint(0, 100, SHAPE).astype(np.uint8)
        tp.locate(dummy_image, 5, engine=self.engine)
        tp.locate(dummy_image, 5, invert=True, engine=self.engine)

        # Check float types
        dummy_image = np.random.rand(*SHAPE)
        tp.locate(dummy_image, 5, engine=self.engine)
        tp.locate(dummy_image, 5, invert=True, engine=self.engine)

    def test_black_image(self):
        self.check_skip()
        black_image = np.zeros((21, 23)).astype(np.uint8)
        warnings.simplefilter('always')
        with assert_produces_warning(UserWarning):
            f = tp.locate(black_image, 5, engine=self.engine, preprocess=False)

    def test_maxima_in_margin(self):
        self.check_skip()
        black_image = np.ones((21, 23)).astype(np.uint8)
        draw_point(black_image, [1, 1], 100)
        with assert_produces_warning(UserWarning):
            f = tp.locate(black_image, 5, engine=self.engine)

    def test_maxima_in_margin_3D(self):
        self.skip_numba()
        black_image = np.ones((21, 23, 25)).astype(np.uint8)
        draw_point(black_image, [1, 1, 1], 100)
        with assert_produces_warning(UserWarning):
            f = tp.locate(black_image, 5, engine=self.engine)

    def test_all_maxima_filtered(self):
        self.check_skip()
        black_image = np.ones((21, 23)).astype(np.uint8)
        draw_point(black_image, [11, 13], 10)
        with assert_produces_warning(UserWarning):
            f = tp.locate(black_image, 5, minmass=1000,
                          engine=self.engine, preprocess=False)

    def test_warn_color_image(self):
        self.check_skip()

        # RGB-like
        image = np.random.randint(0, 100, (21, 23, 3)).astype(np.uint8)
        with assert_produces_warning(UserWarning):
            tp.locate(image, 5)

        # RGBA-like
        image = np.random.randint(0, 100, (21, 23, 4)).astype(np.uint8)
        with assert_produces_warning(UserWarning):
            tp.locate(image, 5)

    def test_flat_peak(self):
        # This tests the part of locate_maxima that eliminates multiple
        # maxima in the same mask area.
        self.check_skip()
        image = np.ones((21, 23)).astype(np.uint8)
        image[11, 13] = 100
        image[11, 14] = 100
        image[12, 13] = 100
        count = len(tp.locate(image, 5, preprocess=False,
                              engine=self.engine))
        self.assertEqual(count, 1)

        image = np.ones((21, 23)).astype(np.uint8)
        image[11:13, 13:15] = 100
        count = len(tp.locate(image, 5, preprocess=False,
                              engine=self.engine))
        self.assertEqual(count, 1)

        image = np.ones((21, 23)).astype(np.uint8)
        image[11, 13] = 100
        image[11, 14] = 100
        image[11, 15] = 100
        count = len(tp.locate(image, 5, preprocess=False,
                              engine=self.engine))
        self.assertEqual(count, 1)

        # This tests that two nearby peaks are merged by
        # picking the one with the brighter neighborhood.
        image = np.ones((21, 23)).astype(np.uint8)
        pos = [14, 14]

        draw_point(image, [11, 13], 100)
        draw_point(image, [11, 14], 100)
        draw_point(image, [11, 15], 100)
        draw_point(image, [14, 13], 101)
        draw_point(image, [14, 14], 101)
        draw_point(image, [14, 15], 101)
        cols = ['y', 'x']
        actual = tp.locate(image, 5, preprocess=False,
                           engine=self.engine)[cols]

        expected = DataFrame(np.asarray(pos).reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=0.1)

        # Break ties by sorting by position, simply to avoid
        # any randomness resulting from cKDTree returning a set.
        image = np.ones((21, 23)).astype(np.uint8)
        pos = [14, 14]

        draw_point(image, [11, 12], 100)
        draw_point(image, [11, 13], 100)
        draw_point(image, [11, 14], 100)
        draw_point(image, [14, 13], 100)
        draw_point(image, [14, 14], 100)
        draw_point(image, [14, 15], 100)
        cols = ['y', 'x']
        actual = tp.locate(image, 5, preprocess=False,
                           engine=self.engine)[cols]

        expected = DataFrame(np.asarray(pos).reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=0.1)

    def test_one_centered_gaussian(self):
        self.check_skip()
        L = 21
        dims = (L, L + 2)  # avoid square images in tests
        pos = [7, 13]
        cols = ['y', 'x']
        expected = DataFrame(np.asarray(pos).reshape(1, -1), columns=cols)

        image = np.ones(dims, dtype='uint8')
        draw_feature(image, pos, 27)
        actual = tp.locate(image, 9, 1, preprocess=False,
                           engine=self.engine)[cols]
        assert_allclose(actual, expected, atol=0.1)

    def test_one_centered_gaussian_3D(self):
        self.skip_numba()
        L = 21
        dims = (L, L + 2, L + 4)  # avoid square images in tests
        pos = [7, 13, 9]
        cols = ['z', 'y', 'x']
        expected = DataFrame(np.asarray(pos).reshape(1, -1), columns=cols)

        image = np.ones(dims, dtype='uint8')
        draw_feature(image, pos, 27)
        actual = tp.locate(image, 9, 1, preprocess=False,
                           engine=self.engine)[cols]
        assert_allclose(actual, expected, atol=0.1)

    def test_one_centered_gaussian_3D_anisotropic(self):
        self.skip_numba()
        L = 21
        dims = (L, L + 2, L + 4)  # avoid square images in tests
        pos = [7, 13, 9]
        cols = ['z', 'y', 'x']
        expected = DataFrame(np.asarray(pos).reshape(1, -1), columns=cols)

        image = np.ones(dims, dtype='uint8')
        draw_feature(image, pos, (21, 27, 27))
        actual = tp.locate(image, (7, 9, 9), 1, preprocess=False,
                           engine=self.engine)[cols]
        assert_allclose(actual, expected, atol=0.1)

    def test_reject_peaks_near_edge(self):
        """Check whether local maxima near the edge of the image
        are properly rejected, so as not to cause crashes later."""
        N = 30
        diameter = 9
        y = np.arange(20, 10*N + 1, 20)
        x = np.linspace(diameter / 2. - 2, diameter * 1.5, len(y))
        cols = ['y', 'x']
        expected = DataFrame(np.vstack([y, x]).T, columns=cols)

        dims = (y.max() - y.min() + 5*diameter, int(4 * diameter) - 2)
        image = np.ones(dims, dtype='uint8')
        for ypos, xpos in expected[['y', 'x']].values:
            draw_feature(image, [ypos, xpos], 27, max_value=100)
        def locate(image, **kwargs):
            return tp.locate(image, diameter, 1, preprocess=False,
                             engine=self.engine, **kwargs)[cols]
        # All we care about is that this doesn't crash
        actual = locate(image)
        assert len(actual)
        # Check the other sides
        actual = locate(np.fliplr(image))
        assert len(actual)
        actual = locate(image.transpose())
        assert len(actual)
        actual = locate(np.flipud(image.transpose()))
        assert len(actual)

    def test_subpx_precision(self): 
        self.check_skip()
        L = 21
        dims = (L, L + 2)  # avoid square images in tests
        cols = ['y', 'x']
        PRECISION = 0.1

        # one bright pixel
        pos = [7, 13]
        image = np.ones(dims, dtype='uint8')
        draw_point(image, pos, 100)
        actual = tp.locate(image, 3, 1, preprocess=False,
                           engine=self.engine)[cols]
        expected = DataFrame(np.asarray(pos).reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=0.001)  # especially precise

        # two neighboring pixels of equal brightness
        pos1 = np.array([7, 13])
        pos2 = np.array([8, 13])
        pos = [7.5, 13]  # center is between pixels
        image = np.ones(dims, dtype='uint8')
        draw_point(image, pos1, 100)
        draw_point(image, pos2, 100)
        actual = tp.locate(image, 5, 1, preprocess=False,
                           engine=self.engine)[cols]
        expected = DataFrame(np.asarray(pos).reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        # two neighboring pixels of unequal brightness
        pos1 = np.array([7, 13])
        pos2 = np.array([8, 13])
        pos = [7.25, 13]  # center is between pixels, biased left
        image = np.ones(dims, dtype='uint8')
        draw_point(image, pos1, 100)
        draw_point(image, pos2, 50)
        actual = tp.locate(image, 5, 1, preprocess=False,
                           engine=self.engine)[cols]
        expected = DataFrame(np.asarray(pos).reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        pos = [7.75, 13]  # center is between pixels, biased right 
        image = np.ones(dims, dtype='uint8')
        draw_point(image, pos1, 50)
        draw_point(image, pos2, 100)
        actual = tp.locate(image, 5, 1, preprocess=False,
                           engine=self.engine)[cols]
        expected = DataFrame(np.asarray(pos).reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        pos1 = np.array([7, 12])
        pos2 = np.array([7, 13])
        pos = [7, 12.25]  # center is between pixels, biased down
        image = np.ones(dims, dtype='uint8')
        draw_point(image, pos1, 100)
        draw_point(image, pos2, 50)
        actual = tp.locate(image, 5, 1, preprocess=False,
                           engine=self.engine)[cols]
        expected = DataFrame(np.asarray(pos).reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        pos = [7, 12.75]  # center is between pixels, biased up 
        image = np.ones(dims, dtype='uint8')
        draw_point(image, pos1, 50)
        draw_point(image, pos2, 100)
        actual = tp.locate(image, 5, 1, preprocess=False,
                           engine=self.engine)[cols]
        expected = DataFrame(np.asarray(pos).reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        # four neighboring pixels of unequal brightness
        pos1 = np.array([7, 13])
        pos2 = np.array([8, 13])
        pos3 = np.array([7, 13])
        pos4 = np.array([8, 13])
        pos = [7.25, 13]  # center is between pixels, biased left
        image = np.ones(dims, dtype='uint8')
        draw_point(image, pos1, 100)
        draw_point(image, pos2, 50)
        draw_point(image, pos3, 100)
        draw_point(image, pos4, 50)
        actual = tp.locate(image, 5, 1, preprocess=False,
                           engine=self.engine)[cols]
        expected = DataFrame(np.asarray(pos).reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        pos = [7.75, 13]  # center is between pixels, biased right 
        image = np.ones(dims, dtype='uint8')
        draw_point(image, pos1, 50)
        draw_point(image, pos2, 100)
        draw_point(image, pos3, 50)
        draw_point(image, pos4, 100)
        actual = tp.locate(image, 5, 1, preprocess=False,
                           engine=self.engine)[cols]
        expected = DataFrame(np.asarray(pos).reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        pos1 = np.array([7, 12])
        pos2 = np.array([7, 13])
        pos3 = np.array([7, 12])
        pos4 = np.array([7, 13])
        pos = [7, 12.25]  # center is between pixels, biased down
        image = np.ones(dims, dtype='uint8')
        draw_point(image, pos1, 100)
        draw_point(image, pos2, 50)
        actual = tp.locate(image, 5, 1, preprocess=False,
                           engine=self.engine)[cols]
        expected = DataFrame(np.asarray(pos).reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        pos = [7, 12.75]  # center is between pixels, biased up 
        image = np.ones(dims, dtype='uint8')
        draw_point(image, pos1, 50)
        draw_point(image, pos2, 100)
        actual = tp.locate(image, 5, 1, preprocess=False,
                           engine=self.engine)[cols]
        expected = DataFrame(np.asarray(pos).reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

    def test_multiple_simple_sparse(self):
        self.check_skip()
        actual, expected = compare((200, 300), 4, 2, noise_level=0,
                                   engine=self.engine)
        assert_allclose(actual, expected, atol=0.5)

    def test_multiple_noisy_sparse(self):
        self.check_skip()
        actual, expected = compare((200, 300), 4, 2, noise_level=1,
                                   engine=self.engine)
        assert_allclose(actual, expected, atol=0.5)

    def test_multiple_more_noisy_sparse(self):
        self.check_skip()
        actual, expected = compare((200, 300), 4, 2, noise_level=2,
                                   engine=self.engine)
        assert_allclose(actual, expected, atol=0.5)

    def test_multiple_anisotropic_3D_simple(self):
        self.skip_numba()
        actual, expected = compare((100, 120, 10), 4, (4, 4, 2), noise_level=0,
                                   engine=self.engine)
        assert_allclose(actual, expected, atol=0.5)

    def test_topn(self):
        self.check_skip()
        L = 21
        dims = (L, L + 2)  # avoid square images in tests
        cols = ['y', 'x']
        PRECISION = 0.1

        # top 2
        pos1 = np.array([7, 7])
        pos2 = np.array([14, 14])
        pos3 = np.array([7, 14])
        image = np.ones(dims, dtype='uint8')
        draw_point(image, pos1, 100)
        draw_point(image, pos2, 90)
        draw_point(image, pos3, 80)
        actual = tp.locate(image, 5, 1, topn=2, preprocess=False,
                           engine=self.engine)[cols]
        actual = actual.sort(['x', 'y'])  # sort for reliable comparison
        expected = DataFrame([pos1, pos2], columns=cols).sort(['x', 'y'])
        assert_allclose(actual, expected, atol=PRECISION)

        # top 1
        actual = tp.locate(image, 5, 1, topn=1, preprocess=False,
                           engine=self.engine)[cols]
        actual = actual.sort(['x', 'y'])  # sort for reliable comparison
        expected = DataFrame([pos1], columns=cols).sort(['x', 'y'])
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
        pos = [50, 55]
        cols = ['y', 'x']

        SIZE = 2
        image = np.ones(dims, dtype='uint8')
        draw_feature(image, pos, (SIZE*2 + 1) * 3)
        actual = tp.locate(image, SIZE*2 + 5, 1, preprocess=False,
                           engine=self.engine)['size']
        expected = SIZE
        assert_allclose(actual, expected, rtol=0.1)

        SIZE = 3
        image = np.ones(dims, dtype='uint8')
        draw_feature(image, pos, (SIZE*2 + 1) * 3)
        actual = tp.locate(image, SIZE*2 + 5, 1, preprocess=False,
                           engine=self.engine)['size']
        expected = SIZE
        assert_allclose(actual, expected, rtol=0.1)

        SIZE = 5
        image = np.ones(dims, dtype='uint8')
        draw_feature(image, pos, (SIZE*2 + 1) * 3)
        actual = tp.locate(image, SIZE*2 + 7, 1, preprocess=False,
                           engine=self.engine)['size']
        expected = SIZE
        assert_allclose(actual, expected, rtol=0.1)

        SIZE = 7
        image = np.ones(dims, dtype='uint8')
        draw_feature(image, pos, (SIZE*2 + 1) * 3)
        actual = tp.locate(image, SIZE*2 + 11, 1, preprocess=False,
                           engine=self.engine)['size']
        expected = SIZE
        assert_allclose(actual, expected, rtol=0.1)

    def test_eccentricity(self):
        # Eccentricity (elongation) is measured with good accuracy and
        # ~0.02 precision, as long as the mask is large enough to cover
        # the whole object.
        self.check_skip()
        L = 501 
        dims = (L + 2, L)  # avoid square images in tests
        pos = [50, 55]
        cols = ['y', 'x']

        ECC = 0
        image = np.ones(dims, dtype='uint8')
        draw_feature(image, pos, 27, ecc=ECC)
        actual = tp.locate(image, 21, 1, preprocess=False,
                           engine=self.engine)['ecc']
        expected = ECC
        assert_allclose(actual, expected, atol=0.02)

        ECC = 0.2
        image = np.ones(dims, dtype='uint8')
        draw_feature(image, pos, 27, ecc=ECC)
        actual = tp.locate(image, 21, 1, preprocess=False,
                           engine=self.engine)['ecc']
        expected = ECC
        assert_allclose(actual, expected, atol=0.1)

        ECC = 0.5
        image = np.ones(dims, dtype='uint8')
        draw_feature(image, pos, 27, ecc=ECC)
        actual = tp.locate(image, 21, 1, preprocess=False,
                           engine=self.engine)['ecc']
        expected = ECC
        assert_allclose(actual, expected, atol=0.1)


    def test_whole_pixel_shifts(self):
        self.check_skip()
        L = 21
        dims = (L, L + 2)  # avoid square images in tests
        pos = [7, 13]
        expected = np.array([pos])

        image = np.ones(dims, dtype='uint8')
        draw_feature(image, pos, 15)

        guess = np.array([[6, 13]])
        actual = tp.feature.refine(image, image, 6, guess, characterize=False,
                                   engine=self.engine)[:, :2][:, ::-1]
        assert_allclose(actual, expected, atol=0.1)

        guess = np.array([[7, 12]])
        actual = tp.feature.refine(image, image, 6, guess, characterize=False,
                                   engine=self.engine)[:, :2][:, ::-1]
        assert_allclose(actual, expected, atol=0.1)

        guess = np.array([[7, 14]])
        actual = tp.feature.refine(image, image, 6, guess, characterize=False,
                                   engine=self.engine)[:, :2][:, ::-1]
        assert_allclose(actual, expected, atol=0.1)

        guess = np.array([[6, 12]])
        actual = tp.feature.refine(image, image, 6, guess, characterize=False,
                                   engine=self.engine)[:, :2][:, ::-1]
        assert_allclose(actual, expected, atol=0.1)


class TestFeatureIdentificationWithVanillaNumpy(
    CommonFeatureIdentificationTests, unittest.TestCase):

    def setUp(self):
        self.engine = 'python'


class TestFeatureIdentificationWithNumba(
    CommonFeatureIdentificationTests, unittest.TestCase):

    def setUp(self):
        self.engine = 'numba'

    def check_skip(self):
        if not NUMBA_AVAILABLE:
            raise nose.SkipTest("Numba not installed. Skipping.")

    def skip_numba(self):
        raise nose.SkipTest("This feature is not "
                            "supported by the numba variant. Skipping.")


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
