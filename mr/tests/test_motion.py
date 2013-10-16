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

class TestDrift(unittest.TestCase):

    def setUp(self):
        N = 10
        Y = 1
        a = DataFrame({'x': np.zeros(N), 'y': np.zeros(N), 
                      'frame': np.arange(N), 'probe': np.zeros(N)})
        b = DataFrame({'x': np.zeros(N - 1), 'y': Y + np.zeros(N - 1), 
                       'frame': np.arange(1, N), 'probe': np.ones(N - 1)})
        self.dead_still = pd.concat([a, b]).reset_index(drop=True)

        P = 1000 # probes
        A = 0.00001 # step amplitude
        np.random.seed(0)
        self.many_walks = pd.concat([DataFrame({'x': A*random_walk(N), 'y': A*random_walk(N), 
            'frame': np.arange(N), 'probe': i}) for i in range(P)]).reset_index(drop=True)

        a = DataFrame({'x': np.arange(N), 'y': np.zeros(N), 
                      'frame': np.arange(N), 'probe': np.zeros(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.zeros(N - 1), 
                       'frame': np.arange(1, N), 'probe': np.ones(N - 1)})
        self.steppers = pd.concat([a, b]).reset_index(drop=True)

    def test_no_drift(self):
        N = 10
        expected = DataFrame({'x': np.zeros(N), 'y': np.zeros(N)}).iloc[1:]
        expected = expected.astype('float')
        expected.index.name = 'frame'
        expected.columns = ['x', 'y']
        # ^ no drift measured for Frame 0

        actual = mr.compute_drift(self.dead_still)
        assert_frame_equal(actual, expected)

        # Small random drift
        actual = mr.compute_drift(self.many_walks)
        assert_frame_equal(actual, expected)

    def test_constant_drift(self):
        N = 10
        expected = DataFrame({'x': np.arange(N), 'y': np.zeros(N)}).iloc[1:]
        expected = expected.astype('float')
        expected.index.name = 'frame'
        expected.columns = ['x', 'y']

        actual = mr.compute_drift(self.steppers)
        assert_frame_equal(actual, expected)

    def test_subtract_zero_drift(self):
        N = 10
        drift = DataFrame(np.zeros((N, 2)), index=np.arange(1, 1 + N))
        drift.columns = ['x', 'y']
        drift.index.name = 'frame'
        actual = mr.subtract_drift(self.dead_still, drift)
        actual = mr.subtract_drift(self.many_walks, drift)
        actual = mr.subtract_drift(self.steppers, drift)
