"""Fit functions to data."""
import unittest
from numpy.testing import assert_allclose

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from mr.core.fitting import fit # Sue me.

class TestFit(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        # Five columns with range 0 through 100 plus Gaussian noise.
        self.x = DataFrame(np.meshgrid(np.ones(5), np.arange(100))[1])
        self.noise = DataFrame(np.random.randn(100, 5))
        # with index of floating-point numbers:
        self.float_idx = self.x.set_index(
            self.x.index.values.astype(np.floating))
        # with an unordered index:
        self.shuffled_idx = self.x.ix[np.random.permutation(self.x.index)]
        self.samples = [self.x, self.float_idx, self.shuffled_idx]

    def test_parabola(self):
        NOISY = 0.1
        func = lambda x, a, b, c: a*x**2 + b*x + c
        index = ['a', 'b', 'c']
        np.random.seed(0)
        true_params = Series(np.random.randn(len(index)), index=index)
        guess = Series(np.random.randn(len(index)), index=index)
        for x in self.samples: 
            y = x.apply(lambda x: func(x, *true_params)) + NOISY*self.noise 
            best_params = fit(y, func, guess)
            assert_allclose([true_params['a']]*len(best_params),
                            best_params['a'], rtol=0.001)
            assert_allclose([true_params['b']]*len(best_params),
                            best_params['b'], rtol=0.01)

    def test_decaying_exp(self):
        NOISY = 0.001
        func = lambda x, a: np.exp(-x/a)
        index = ['a']
        np.random.seed(0)
        true_params = Series(np.random.randn(len(index)), index=index)
        guess = Series(np.random.randn(len(index)), index=index)
        for x in self.samples:
            y = x.apply(lambda x: func(x, *true_params)) + NOISY*self.noise
            best_params = fit(y, func, guess)
            assert_allclose([true_params['a']]*len(best_params),
                            best_params['a'], rtol=0.01)

    def test_gaussian(self):
        NOISY = 0.01
        func = lambda x, a, b: a*np.exp(-x**2/b)
        index = ['a', 'b']
        np.random.seed(0)
        true_params = Series(np.random.randn(len(index)), index=index)
        guess = Series(np.random.randn(len(index)), index=index)
        for x in self.samples:
            y = x.apply(lambda x: func(x, *true_params)) + NOISY*self.noise
            best_params = fit(y, func, guess)
            assert_allclose([true_params['a']]*len(best_params),
                            best_params['a'], rtol=0.1)
            assert_allclose([true_params['b']]*len(best_params),
                            best_params['b'], rtol=0.05)

    def test_power(self):
        NOISY = 0.1
        func = lambda x, a, b, c: a*x**b + c
        index = ['a', 'b', 'c']
        np.random.seed(0)
        true_params = Series(np.random.randn(len(index)), index=index)
        guess = Series(np.random.randn(len(index)), index=index)
        for x in self.samples: 
            y = x.apply(lambda x: func(x, *true_params)) + NOISY*self.noise 
            best_params = fit(y, func, guess)
            assert_allclose([true_params['a']]*len(best_params),
                            best_params['a'], rtol=0.1)
            assert_allclose([true_params['b']]*len(best_params),
                            best_params['b'], rtol=0.05)
            assert_allclose([true_params['c']]*len(best_params),
                            best_params['c'], rtol=0.2)

    def test_power_using_log_residual(self):
        NOISY = 0.1
        func = lambda x, a, b: a*x**b
        index = ['a', 'b']
        np.random.seed(0)
        true_params = Series(np.random.randn(len(index)), index=index)
        guess = Series(np.random.randn(len(index)), index=index)
        for x in self.samples: 
            y = x.apply(lambda x: func(x, *true_params)) + NOISY*self.noise 
            best_params = fit(y, func, guess, log_residual=True)
            print true_params
            print best_params
            assert_allclose([true_params['a']]*len(best_params),
                            best_params['a'], rtol=0.1)
            assert_allclose([true_params['b']]*len(best_params),
                            best_params['b'], rtol=0.05)

    def test_nan(self):
        NOISY = 0.1
        func = lambda x, a, b, c: a*x**2 + b*x + c
        index = ['a', 'b', 'c']
        np.random.seed(0)
        true_params = Series(np.random.randn(len(index)), index=index)
        guess = Series(np.random.randn(len(index)), index=index)
        for x in self.samples: 
            y = x.apply(lambda x: func(x, *true_params)) + NOISY*self.noise 
            # Choose three elements are random, and make them NaN.
            y.ix[np.random.randint(0, 100, 3)] = np.nan
            best_params = fit(y, func, guess)
            assert_allclose([true_params['a']]*len(best_params),
                            best_params['a'], rtol=0.001)
            assert_allclose([true_params['b']]*len(best_params),
                            best_params['b'], rtol=0.01)

    def test_inf(self):
        NOISY = 0.1
        func = lambda x, a, b, c: a*x**2 + b*x + c
        index = ['a', 'b', 'c']
        np.random.seed(0)
        true_params = Series(np.random.randn(len(index)), index=index)
        guess = Series(np.random.randn(len(index)), index=index)
        for x in self.samples: 
            y = x.apply(lambda x: func(x, *true_params)) + NOISY*self.noise 
            # Choose three elements are random, and make them inf.
            y.ix[np.random.randint(0, 100, 3)] = np.inf
            best_params = fit(y, func, guess)
            assert_allclose([true_params['a']]*len(best_params),
                            best_params['a'], rtol=0.001)
            assert_allclose([true_params['b']]*len(best_params),
                            best_params['b'], rtol=0.01)

