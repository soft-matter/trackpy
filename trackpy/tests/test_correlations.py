import numpy as np
from numpy.testing import assert_allclose
from pandas import DataFrame
from trackpy.tests.common import StrictTestCase
from trackpy.utils import pandas_concat

import trackpy as tp


class TestCorrelations(StrictTestCase):

    def setUp(self):
        np.random.seed(0)
        randn = np.random.randn
        N = 500
        a = DataFrame(randn(N, 2), columns=['x', 'y'])
        b = DataFrame(a[['x', 'y']] + 0.1*randn(N, 2), columns=['x', 'y'])
        a['particle'] = np.arange(N)
        b['particle'] = np.arange(N)
        a['frame'] = 0
        b['frame'] = 1
        self.random_walk = pandas_concat([a, b])

    def test_no_correlations(self):
        v = tp.velocity_corr(self.random_walk, 0, 1)
        binned = v.groupby(np.digitize(v.r, np.linspace(0, 1, 10))).mean()
        actual = binned['dot_product']
        expected = np.zeros_like(actual)
        assert_allclose(actual, expected, atol=1e-3)


if __name__ == '__main__':
    import unittest
    unittest.main()
