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
from trackpy.locate.brightfield_ring import locate_brightfield_ring

path, _ = os.path.split(os.path.abspath(__file__))

def artificial_image(shape, count, radius, noise_level, dip=False, **kwargs):
    radius = tp.utils.validate_tuple(radius, len(shape))
    # tp.locate ignores a margin of size radius, take 1 px more to be safe
    margin = tuple([r + 1 for r in radius])
    diameter = tuple([(r * 2) + 1 for r in radius])
    size = [d / 2 for d in diameter]
    separation = tuple([d * 3 for d in diameter])
    cols = ['x', 'y', 'z'][:len(shape)][::-1]
    pos = gen_nonoverlapping_locations(shape, count, separation, margin)
    image = draw_features_brightfield(shape, pos, size, noise_level, dip=dip)
    f = locate_brightfield_ring(image, diameter, **kwargs)
    actual = pandas_sort(f[cols], cols)
    expected = pandas_sort(DataFrame(pos, columns=cols), cols)

    if True:
        plot_result(image, actual, expected)

    return actual, expected

def plot_result(image, actual, expected):
    plt.imshow(image, cmap='gray')
    plt.scatter(actual['x'], actual['y'], c='r', label='actual', s=3)
    plt.scatter(expected['x'], expected['y'], c='g', label='expected', s=3)
    plt.legend()
    plt.show(block=True)



class TestLocateBrightfieldRing(StrictTestCase):
    def setUp(self):
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)

    def test_multiple_simple_sparse(self):
        actual, expected = artificial_image((200, 300), 4, 5, noise_level=0)
        assert_allclose(actual, expected, atol=0.5)

    def test_multiple_simple_sparse_dip(self):
        actual, expected = artificial_image((200, 300), 4, 5, noise_level=0,
                                            dip=True)
        assert_allclose(actual, expected, atol=0.5)

    def test_multiple_noisy_sparse(self):
        #  4% noise
        actual, expected = artificial_image((200, 300), 4, 5, noise_level=10)
        assert_allclose(actual, expected, atol=0.5)

    def test_multiple_noisy_sparse_dip(self):
        actual, expected = artificial_image((200, 300), 4, 5, noise_level=10,
                                            dip=True)
        assert_allclose(actual, expected, atol=0.5)

    def test_multiple_more_noisy_sparse(self):
        # 20% noise
        actual, expected = artificial_image((200, 300), 4, 5, noise_level=51)
        assert_allclose(actual, expected, atol=0.5)

    def test_multiple_more_noisy_sparse_dip(self):
        actual, expected = artificial_image((200, 300), 4, 5, noise_level=51,
                                            dip=True)
        assert_allclose(actual, expected, atol=0.5)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
