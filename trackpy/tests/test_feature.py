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
from numpy.testing import (assert_almost_equal, assert_allclose,
                           assert_array_less)
from numpy.testing.decorators import slow
from pandas.util.testing import (assert_series_equal, assert_frame_equal,
                                 assert_produces_warning)

import trackpy as tp
from trackpy.try_numba import NUMBA_AVAILABLE
from trackpy.artificial import (draw_feature, draw_spots, draw_point,
                                gen_nonoverlapping_locations)

from scipy.spatial import cKDTree

# Catch attempts to set values on an inadvertent copy of a Pandas object.
tp.utils.make_pandas_strict()

path, _ = os.path.split(os.path.abspath(__file__))


def compare(shape, count, radius, noise_level, engine):
    radius = tp.utils.validate_tuple(radius, len(shape))
    # tp.locate ignores a margin of size radius, take 1 px more to be safe
    margin = tuple([r + 1 for r in radius])
    diameter = tuple([(r * 2) + 1 for r in radius])
    draw_range = tuple([d * 3 for d in diameter])
    cols = ['x', 'y', 'z'][:len(shape)][::-1]
    pos = gen_nonoverlapping_locations(shape, count, draw_range, margin)
    image = draw_spots(shape, pos, draw_range, noise_level)
    f = tp.locate(image, diameter, engine=engine)
    actual = f[cols].sort_values(by=cols)
    expected = DataFrame(pos, columns=cols).sort_values(by=cols)
    return actual, expected


def sort_positions(actual, expected):
    tree = cKDTree(actual)
    deviations, argsort = tree.query([expected])
    return deviations, actual[argsort][0]


class OldMinmass(unittest.TestCase):
    def check_skip(self):
        pass

    def setUp(self):
        self.shape = (128, 128)
        self.pos = gen_nonoverlapping_locations(self.shape, 10, separation=20,
                                                margin=10)
        self.N = len(self.pos)
        self.draw_diameter = 25
        self.tp_diameter = 15

    def test_oldmass_8bit(self):
        old_minmass = 11000
        im = draw_spots(self.shape, self.pos, self.draw_diameter, bitdepth=8,
                        noise_level=50)

        new_minmass = tp.minmass_version_change(im, old_minmass,
                                                smoothing_size=self.tp_diameter)
        f = tp.locate(im, self.tp_diameter, minmass=new_minmass)
        assert len(f) == self.N

    def test_oldmass_12bit(self):
        old_minmass = 2800000
        im = draw_spots(self.shape, self.pos, self.draw_diameter, bitdepth=12,
                        noise_level=500)

        new_minmass = tp.minmass_version_change(im, old_minmass,
                                                smoothing_size=self.tp_diameter)
        f = tp.locate(im, self.tp_diameter, minmass=new_minmass)
        assert len(f) == self.N

    def test_oldmass_16bit(self):
        old_minmass = 2800000
        im = draw_spots(self.shape, self.pos, self.draw_diameter, bitdepth=16,
                        noise_level=10000)

        new_minmass = tp.minmass_version_change(im, old_minmass,
                                                smoothing_size=self.tp_diameter)
        f = tp.locate(im, self.tp_diameter, minmass=new_minmass)
        assert len(f) == self.N

    def test_oldmass_float(self):
        old_minmass = 5500
        im = draw_spots(self.shape, self.pos, self.draw_diameter, bitdepth=8,
                        noise_level=50)
        im = (im / im.max()).astype(np.float)

        new_minmass = tp.minmass_version_change(im, old_minmass,
                                                smoothing_size=self.tp_diameter)
        f = tp.locate(im, self.tp_diameter, minmass=new_minmass)
        assert len(f) == self.N

    def test_oldmass_invert(self):
        old_minmass = 2800000
        im = draw_spots(self.shape, self.pos, self.draw_diameter, bitdepth=12,
                        noise_level=500)
        im = (im.max() - im + 10000)

        new_minmass = tp.minmass_version_change(im, old_minmass, invert=True,
                                                smoothing_size=self.tp_diameter)
        f = tp.locate(im, self.tp_diameter, minmass=new_minmass, invert=True)
        assert len(f) == self.N


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
        self.check_skip()
        black_image = np.ones((21, 23, 25)).astype(np.uint8)
        draw_point(black_image, [1, 1, 1], 100)
        with assert_produces_warning(UserWarning):
            f = tp.locate(black_image, 5, engine=self.engine)

    def test_all_maxima_filtered(self):
        self.check_skip()
        black_image = np.ones((21, 23)).astype(np.uint8)
        draw_point(black_image, [11, 13], 10)
        with assert_produces_warning(UserWarning):
            f = tp.locate(black_image, 5, minmass=200,
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
        self.check_skip()
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
        self.check_skip()
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
        pos = [7.33, 13]  # center is between pixels, biased left
        image = np.ones(dims, dtype='uint8')
        draw_point(image, pos1, 100)
        draw_point(image, pos2, 50)
        actual = tp.locate(image, 5, 1, preprocess=False,
                           engine=self.engine)[cols]
        expected = DataFrame(np.asarray(pos).reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        pos = [7.67, 13]  # center is between pixels, biased right
        image = np.ones(dims, dtype='uint8')
        draw_point(image, pos1, 50)
        draw_point(image, pos2, 100)
        actual = tp.locate(image, 5, 1, preprocess=False,
                           engine=self.engine)[cols]
        expected = DataFrame(np.asarray(pos).reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        pos1 = np.array([7, 12])
        pos2 = np.array([7, 13])
        pos = [7, 12.33]  # center is between pixels, biased up
        image = np.ones(dims, dtype='uint8')
        draw_point(image, pos1, 100)
        draw_point(image, pos2, 50)
        actual = tp.locate(image, 5, 1, preprocess=False,
                           engine=self.engine)[cols]
        expected = DataFrame(np.asarray(pos).reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        pos = [7, 12.67]  # center is between pixels, biased down
        image = np.ones(dims, dtype='uint8')
        draw_point(image, pos1, 50)
        draw_point(image, pos2, 100)
        actual = tp.locate(image, 5, 1, preprocess=False,
                           engine=self.engine)[cols]
        expected = DataFrame(np.asarray(pos).reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        # four neighboring pixels of unequal brightness
        pos1 = np.array([7, 12])
        pos2 = np.array([8, 12])
        pos3 = np.array([7, 13])
        pos4 = np.array([8, 13])
        pos = [7.33, 12.5]  # center is between pixels, biased left
        image = np.ones(dims, dtype='uint8')
        draw_point(image, pos1, 100)
        draw_point(image, pos2, 50)
        draw_point(image, pos3, 100)
        draw_point(image, pos4, 50)
        actual = tp.locate(image, 5, 1, preprocess=False,
                           engine=self.engine)[cols]
        expected = DataFrame(np.asarray(pos).reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        pos = [7.67, 12.5]  # center is between pixels, biased right
        image = np.ones(dims, dtype='uint8')
        draw_point(image, pos1, 50)
        draw_point(image, pos2, 100)
        draw_point(image, pos3, 50)
        draw_point(image, pos4, 100)
        actual = tp.locate(image, 5, 1, preprocess=False,
                           engine=self.engine)[cols]
        expected = DataFrame(np.asarray(pos).reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        pos = [7.5, 12.33]  # center is between pixels, biased up
        image = np.ones(dims, dtype='uint8')
        draw_point(image, pos1, 100)
        draw_point(image, pos2, 100)
        draw_point(image, pos3, 50)
        draw_point(image, pos4, 50)
        actual = tp.locate(image, 5, 1, preprocess=False,
                           engine=self.engine)[cols]
        expected = DataFrame(np.asarray(pos).reshape(1, -1), columns=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        pos = [7.5, 12.67]  # center is between pixels, biased down
        image = np.ones(dims, dtype='uint8')
        draw_point(image, pos1, 50)
        draw_point(image, pos2, 50)
        draw_point(image, pos3, 100)
        draw_point(image, pos4, 100)
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
        self.check_skip()
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
        actual = actual.sort_values(by=['x', 'y'])  # sort for reliable comparison
        expected = DataFrame([pos1, pos2], columns=cols).sort_values(by=['x', 'y'])
        assert_allclose(actual, expected, atol=PRECISION)

        # top 1
        actual = tp.locate(image, 5, 1, topn=1, preprocess=False,
                           engine=self.engine)[cols]
        actual = actual.sort_values(by=['x', 'y'])  # sort for reliable comparison
        expected = DataFrame([pos1], columns=cols).sort_values(by=['x', 'y'])
        assert_allclose(actual, expected, atol=PRECISION)

    def test_minmass_maxsize(self):
        # Test the mass- and sizebased filtering here on 4 different features.
        self.check_skip()
        L = 64
        dims = (L, L + 2)
        cols = ['y', 'x']
        PRECISION = 1  # we are not testing for subpx precision here

        image = np.zeros(dims, dtype=np.uint8)
        pos1 = np.array([15, 20])
        pos2 = np.array([40, 40])
        pos3 = np.array([25, 45])
        pos4 = np.array([35, 15])

        draw_feature(image, pos1, 15)
        draw_feature(image, pos2, 30)
        draw_feature(image, pos3, 5)
        draw_feature(image, pos4, 20)

        # filter on mass
        actual = tp.locate(image, 15, engine=self.engine, preprocess=False,
                           minmass=6500)[cols]
        actual = actual.sort_values(by=cols)
        expected = DataFrame([pos2, pos4], columns=cols).sort_values(by=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        # filter on size
        actual = tp.locate(image, 15, engine=self.engine, preprocess=False,
                           maxsize=3.0)[cols]
        actual = actual.sort_values(by=cols)
        expected = DataFrame([pos1, pos3], columns=cols).sort_values(by=cols)
        assert_allclose(actual, expected, atol=PRECISION)

        # filter on both mass and size
        actual = tp.locate(image, 15, engine=self.engine, preprocess=False,
                           minmass=600, maxsize=4.0)[cols]
        actual = actual.sort_values(by=cols)
        expected = DataFrame([pos1, pos4], columns=cols).sort_values(by=cols)
        assert_allclose(actual, expected, atol=PRECISION)

    def test_mass(self):
        # The mass calculated from the processed image should be independent
        # of added noise. Its absolute value is untested.

        # The mass calculated from the raw image should equal
        # noiseless mass + noise_size/2 * Npx_in_mask.
        self.check_skip()
        ndim = 2
        radius = 6
        N = 20
        shape = (128, 127)

        # Calculate the expected mass from a single spot using the set masksize
        center = (radius*2,) * ndim
        spot = draw_spots((radius*4,) * ndim, [center], radius*3, bitdepth=12)
        rect = [slice(c - radius, c + radius + 1) for c in center]
        mask = tp.masks.binary_mask(radius, 2)
        Npx = mask.sum()
        EXPECTED_MASS = (spot[rect] * mask).sum()

        # Generate feature locations and make the image
        expected = gen_nonoverlapping_locations(shape, N, radius*3, radius+2)
        expected = expected + np.random.random(expected.shape)
        N = expected.shape[0]
        image = draw_spots(shape, expected, radius*3, bitdepth=12)

        # analyze the image without noise
        f = tp.locate(image, radius*2+1, engine=self.engine, topn=N)
        PROCESSED_MASS = f['mass'].mean()
        assert_allclose(f['raw_mass'].mean(), EXPECTED_MASS, rtol=0.01)

        for n, noise in enumerate(np.arange(0.05, 0.8, 0.05)):
            noise_level = int((2**12 - 1) * noise)
            image_noisy = image + np.array(np.random.randint(0, noise_level,
                                                             image.shape),
                                           dtype=image.dtype)
            f = tp.locate(image_noisy, radius*2+1, engine=self.engine, topn=N)
            assert_allclose(f['mass'].mean(), PROCESSED_MASS, rtol=0.1)
            assert_allclose(f['raw_mass'].mean(),
                            EXPECTED_MASS + Npx*noise_level/2, rtol=0.1)

    def test_size(self):
        # To draw Gaussians with radii 2, 3, 5, and 7 px, we supply the draw
        # function with rg=0.25. This means that the radius of gyration will be
        # one fourth of the max radius in the draw area, which is diameter/2.

        # The 'size' comes out to be within 3%, which is because of the
        # pixelation of the Gaussian.

        # The IDL code has mistake in this area, documented here:
        # http://www.physics.emory.edu/~weeks/idl/radius.html

        self.check_skip()
        L = 101
        dims = (L, L + 2)  # avoid square images in tests
        for pos in [[50, 55], [50.2, 55], [50.5, 55]]:
            for SIZE in [2, 3, 5, 7]:
                image = np.zeros(dims, dtype='uint8')
                draw_feature(image, pos, SIZE*8, rg=0.25)
                actual = tp.locate(image, SIZE*8 - 1, 1, preprocess=False,
                                   engine=self.engine)['size']
                assert_allclose(actual, SIZE, rtol=0.1)

    def test_size_anisotropic(self):
        # The separate columns 'size_x' and 'size_y' reflect the radii of
        # gyration in the two separate directions.

        self.check_skip()
        L = 101
        SIZE = 5
        dims = (L, L + 2)  # avoid square images in tests
        pos = [50, 55]
        for ar in [1.1, 1.5, 2]:
            image = np.zeros(dims, dtype='uint8')
            draw_feature(image, pos, [int(SIZE*8*ar), SIZE*8], rg=0.25)
            f = tp.locate(image, [int(SIZE*4*ar) * 2 - 1, SIZE*8 - 1], 1,
                          preprocess=False, engine=self.engine)
            assert_allclose(f['size_x'], SIZE, rtol=0.1)
            assert_allclose(f['size_y'], SIZE*ar, rtol=0.1)

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

    def test_ep(self):
        # Test whether the estimated static error equals the rms deviation from
        # the expected values. Next to the feature mass, the static error is
        # calculated from the estimated image background level and variance.
        # This estimate is also tested here.

        # A threshold is necessary to identify the background array so that
        # background average and standard deviation can be estimated within 1%
        # accuracy.

        # The (absolute) tolerance for ep in this test is 0.05 pixels.
        # Parameters are tweaked so that there is no deviation due to a too
        # small mask size. Signal/noise ratios up to 50% are tested.
        self.check_skip()
        draw_diameter = 21
        locate_diameter = 15
        N = 200
        shape = (512, 513)
        noise_expectation = np.array([1/2., np.sqrt(1/12.)])  # average, stdev

        # Generate feature locations and make the image
        expected = gen_nonoverlapping_locations(shape, N, draw_diameter,
                                                locate_diameter)
        expected = expected + np.random.random(expected.shape)
        N = expected.shape[0]
        image = draw_spots(shape, expected, draw_diameter, bitdepth=12)

        for n, noise in enumerate([0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]):
            noise_level = int((2**12 - 1) * noise)
            image_noisy = image + np.array(np.random.randint(0, noise_level,
                                                             image.shape),
                                           dtype=image.dtype)

            f = tp.locate(image_noisy, locate_diameter, engine=self.engine,
                          topn=N, threshold=noise_level/4)

            _, actual = sort_positions(f[['y', 'x']].values, expected)
            rms_dev = np.sqrt(np.mean(np.sum((actual-expected)**2, 1)))
            assert_allclose(rms_dev, f['ep'].mean(), atol=0.05)

            # Additionally test the measured noise
            actual_noise = tp.uncertainty.measure_noise(image, image_noisy,
                                                        locate_diameter // 2)
            assert_allclose(actual_noise, noise_expectation * noise_level,
                            rtol=0.01, atol=1)

    def test_ep_anisotropic(self):
        # The separate columns 'ep_x' and 'ep_y' reflect the static errors
        # in the two separate directions. The error in the direction with the
        # smallest mask size should be lowest; their ratio is equal to the
        # mask aspect ratio.

        self.check_skip()
        L = 101
        SIZE = 5
        dims = (L, L + 2)  # avoid square images in tests
        pos = [50, 55]
        noise = 0.2
        for ar in [1.1, 1.5, 2]:  # sizeY / sizeX
            image = np.random.randint(0, int(noise*255), dims).astype('uint8')
            draw_feature(image, pos, [int(SIZE*8*ar), SIZE*8],
                         max_value=int((1-noise)*255))
            f = tp.locate(image, [int(SIZE*4*ar) * 2 - 1, SIZE*8 - 1],
                          threshold=int(noise*64), topn=1,
                          engine=self.engine).loc[0]
            assert_allclose(f['ep_y'] / f['ep_x'], ar, rtol=0.1)

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

    def test_uncertainty_failure(self):
        """When the uncertainty ("ep") calculation results in a nonsense negative
        value, it should return NaN instead.
        """
        self.check_skip()
        L = 21
        dims = (L, L + 2)  # avoid square images in tests
        pos = np.array([7, 13])
        cols = ['x', 'y']
        expected = DataFrame(pos[::-1].reshape(1, -1), columns=cols)

        image = 100*np.ones(dims, dtype='uint8')
        # For a feature to have a negative uncertainty, its integrated mass
        # must be less than if it were not there at all (replaced
        # with the average background intensity). So our feature will be a
        # small bright spot surrounded by a dark annulus.
        draw_feature(image, pos, 6, max_value=-100)
        draw_feature(image, pos, 4, max_value=200)

        actual = tp.locate(image, 9, 1, preprocess=False, engine=self.engine)
        assert np.allclose(actual[['x', 'y']], expected[['x', 'y']])
        assert np.isnan(np.asscalar(actual.ep))


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
