from __future__ import division
import mr
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import unittest
import nose
from numpy.testing import assert_almost_equal, assert_allclose
from numpy.testing.decorators import slow
from pandas.util.testing import (assert_series_equal, assert_frame_equal)

class TestTracking(unittest.TestCase):

    def setUp(self):
        pass

    def test_one_trivial(self):
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.zeros(N), 'frame': np.arange(N)})
        actual = mr.track(f)
        expected = f.copy()
        expected['probe'] = np.zeros(N)
        assert_frame_equal(actual, expected)

    def test_two_trivial(self):
        N = 5
        Y = 2
        a = DataFrame({'x': np.arange(N), 'y': np.zeros(N), 'frame': np.arange(N)})
        b = DataFrame({'x': np.arange(N), 'y': Y*np.ones(N), 'frame': np.arange(N)})
        f = pd.concat([a, b])
        actual = mr.track(f)
        expected = f.copy().reset_index(drop=True)
        expected['probe'] = np.concatenate([np.zeros(N), np.ones(N)])
        assert_frame_equal(actual, expected)

        # Shuffle rows
#        f.reindex(np.random.permutation(f.index))
#        actual = mr.track(f)
#        print actual
#        assert_frame_equal(actual, expected)

