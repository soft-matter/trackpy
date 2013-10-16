from __future__ import division
import mr
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import unittest
import nose
from numpy.testing import assert_almost_equal, assert_allclose
from numpy.testing.decorators import slow
from pandas.util.testing import (assert_series_equal, assert_frame_equal,
                                 assert_almost_equal)

def random_walk(N):
    return np.cumsum(np.random.randn(N))

class CommonTrackingTests(object):

    def test_one_trivial_stepper(self):
        # One 1D stepper
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.zeros(N), 'frame': np.arange(N)})
        actual = self.track(f, 5)
        expected = f.copy()
        expected['probe'] = np.zeros(N)
        assert_frame_equal(actual, expected)

    def test_two_trivial_steppers(self):
        N = 5
        Y = 2
        # Begin second feature one frame later than the first, so the probe labeling (0, 1) is
        # established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.zeros(N), 'frame': np.arange(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.zeros(N - 1), 'frame': np.arange(1, N)})
        f = pd.concat([a, b])
        actual = self.track(f, 5)
        expected = f.copy().reset_index(drop=True)
        expected['probe'] = np.concatenate([np.zeros(N), np.ones(N - 1)])
        expected.sort(['probe', 'frame'], inplace=True)
        assert_frame_equal(actual, expected)

        # Sort rows by frame (normal use)
        actual = self.track(f.sort('frame'), 5)
        assert_frame_equal(actual, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.track(f1)
        assert_frame_equal(actual, expected)

    def test_isolated_continuous_random_walks(self):
        # Two 2D random walks
        np.random.seed(0)
        N = 30
        Y = 10
        M = 20 # margin, because negative values raise OutOfHash
        a = DataFrame({'x': M + random_walk(N), 'y': M + random_walk(N), 'frame': np.arange(N)})
        b = DataFrame({'x': M + random_walk(N - 1), 'y': M + Y + random_walk(N - 1), 'frame': np.arange(1, N)})
        f = pd.concat([a, b])
        actual = self.track(f, 5)
        expected = f.copy().reset_index(drop=True)
        expected['probe'] = np.concatenate([np.zeros(N), np.ones(N - 1)])
        expected.sort(['probe', 'frame'], inplace=True)
        assert_frame_equal(actual, expected)

        # Many 2D random walks
        np.random.seed(0)
        initial_positions = [(10, 11), (10, 18), (14, 15), (20, 21)]
        import itertools
        c = itertools.count()
        def walk(x, y): 
            i = next(c)
            return DataFrame({'x': x + random_walk(N - i), 'y': y + random_walk(N - i),
                             'frame': np.arange(i, N)})
        f = pd.concat([walk(*pos) for pos in initial_positions])
        actual = self.track(f, 5)
        expected = f.copy().reset_index(drop=True)
        expected['probe'] = np.concatenate([i*np.ones(N - i) for i in range(len(initial_positions))])
        expected.sort(['probe', 'frame'], inplace=True)
        assert_frame_equal(actual, expected)

# for the future
# class TestKDTreeTracking(CommonTrackingTests, unittest.TestCase):
#    def setUp(self):
#        self.track = mr.core.tracking.kdtree_track

class TestCaswellTracking(CommonTrackingTests, unittest.TestCase):
    def setUp(self):
        self.track = mr.track

