import unittest
import nose
from numpy.testing import assert_allclose

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import mr

class TestCorrelations(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        randn = np.random.randn
        N = 500
        a = DataFrame(randn(N, 2), columns=['x', 'y'])
        b = DataFrame(a[['x', 'y']] + 0.1*randn(N, 2), columns=['x', 'y'])
        a['probe'] = np.arange(N)
        b['probe'] = np.arange(N)
        a['frame'] = 0
        b['frame'] = 1
        self.random_walk = pd.concat([a, b])

    def test_no_correlations(self):
        v = mr.velocity_corr(self.random_walk, 0, 1)
        binned = v.groupby(np.digitize(v.r, np.linspace(0, 1, 10))).mean()
        actual = binned['dot_product']
        expected = np.zeros_like(actual)
        assert_allclose(actual, expected, atol=1e-3)

