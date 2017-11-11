from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas.util.testing import (assert_series_equal, assert_frame_equal,
                                 assert_almost_equal)

import trackpy as tp
from trackpy.utils import pandas_sort
from trackpy.tests.common import StrictTestCase

def random_walk(N):
    return np.cumsum(np.random.randn(N))


def conformity(df):
    """ Organize toy data to look like real data. Be strict about dtypes:
    particle is a float and frame is an integer."""
    df['frame'] = df['frame'].astype(np.int)
    df['particle'] = df['particle'].astype(np.float)
    df['x'] = df['x'].astype(np.float)
    df['y'] = df['y'].astype(np.float)
    df.set_index('frame', drop=False, inplace=True)
    return pandas_sort(df, by=['frame', 'particle'])


def assert_traj_equal(t1, t2):
    return assert_frame_equal(conformity(t1), conformity(t2))


def add_drift(df, drift):
    df = df.copy()
    df['x'] = df['x'].add(drift['x'], fill_value=0)
    df['y'] = df['y'].add(drift['y'], fill_value=0)
    return df


class TestDrift(StrictTestCase):
    def setUp(self):
        N = 10
        Y = 1
        a = DataFrame({'x': np.zeros(N), 'y': np.zeros(N),
                       'frame': np.arange(N), 'particle': np.zeros(N)})
        b = DataFrame({'x': np.zeros(N - 1), 'y': Y + np.zeros(N - 1),
                       'frame': np.arange(1, N), 'particle': np.ones(N - 1)})
        self.dead_still = conformity(pd.concat([a, b]))
        pandas_sort(self.dead_still, ['frame', 'particle'], inplace=True)

        P = 1000 # particles
        A = 0.00001 # step amplitude
        np.random.seed(0)
        particles = [DataFrame({'x': A*random_walk(N), 'y': A*random_walk(N),
                                'frame': np.arange(N), 'particle': i})
                     for i in range(P)]
        self.many_walks = conformity(pd.concat(particles))

        a = DataFrame({'x': np.arange(N), 'y': np.zeros(N),
                       'frame': np.arange(N), 'particle': np.zeros(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.zeros(N - 1),
                       'frame': np.arange(1, N), 'particle': np.ones(N - 1)})
        self.steppers = conformity(pd.concat([a, b]))

    def test_no_drift(self):
        N = 10
        expected = DataFrame({'x': np.zeros(N), 'y': np.zeros(N)}).iloc[1:]
        expected = expected.astype('float')
        expected.index.name = 'frame'
        expected.columns = ['x', 'y']
        # ^ no drift measured for Frame 0

        actual = tp.compute_drift(self.dead_still)
        assert_frame_equal(actual, expected[['y', 'x']])

        actual_rolling = tp.compute_drift(self.dead_still,3)
        assert_frame_equal(actual_rolling, expected[['y', 'x']])

        # Small random drift
        actual = tp.compute_drift(self.many_walks)
        assert_frame_equal(actual, expected[['y', 'x']])

    def test_constant_drift(self):
        N = 10
        expected = DataFrame({'x': np.arange(N), 'y': np.zeros(N)}).iloc[1:]
        expected = expected.astype('float')
        expected.index.name = 'frame'
        expected.columns = ['x', 'y']

        actual = tp.compute_drift(self.steppers)
        assert_frame_equal(actual, expected[['y', 'x']])

    def test_subtract_zero_drift(self):
        N = 10
        drift = DataFrame(np.zeros((N - 1, 2)),
                          np.arange(1, N, dtype=np.int)).astype('float64')
        drift.columns = ['x', 'y']
        drift.index.name = 'frame'
        actual = tp.subtract_drift(self.dead_still, drift)
        assert_traj_equal(actual, self.dead_still)
        actual = tp.subtract_drift(self.many_walks, drift)
        assert_traj_equal(actual, self.many_walks)
        actual = tp.subtract_drift(self.steppers, drift)
        assert_traj_equal(actual, self.steppers)

    def test_subtract_constant_drift(self):
        N = 10
        # Add a constant drift here, and then use subtract_drift to
        # subtract it.
        drift = DataFrame(np.outer(np.arange(N - 1), [1, 1]),
                          index=np.arange(1, N, dtype=np.int)).astype('float64')
        drift.columns = ['x', 'y']
        drift.index.name = 'frame'
        actual = tp.subtract_drift(add_drift(self.dead_still, drift), drift)
        assert_traj_equal(actual, self.dead_still)
        actual = tp.subtract_drift(add_drift(self.many_walks, drift), drift)
        assert_traj_equal(actual, self.many_walks)
        actual = tp.subtract_drift(add_drift(self.steppers, drift), drift)
        assert_traj_equal(actual, self.steppers)


class TestMSD(StrictTestCase):
    def setUp(self):
        N = 10
        Y = 1
        a = DataFrame({'x': np.zeros(N), 'y': np.zeros(N),
                      'frame': np.arange(N), 'particle': np.zeros(N)})
        b = DataFrame({'x': np.zeros(N - 1), 'y': Y + np.zeros(N - 1),
                       'frame': np.arange(1, N), 'particle': np.ones(N - 1)})
        self.dead_still = conformity(pd.concat([a, b]))

        P = 50 # particles
        A = 1 # step amplitude
        np.random.seed(0)
        particles = [DataFrame({'x': A*random_walk(N), 'y': A*random_walk(N),
                                'frame': np.arange(N), 'particle': i})
                     for i in range(P)]
        self.many_walks = conformity(pd.concat(particles))

        a = DataFrame({'x': np.arange(N), 'y': np.zeros(N),
                      'frame': np.arange(N), 'particle': np.zeros(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.zeros(N - 1),
                       'frame': np.arange(1, N), 'particle': np.ones(N - 1)})
        self.steppers = conformity(pd.concat([a, b]))

    def test_zero_emsd(self):
        N = 10
        actual = tp.emsd(self.dead_still, 1, 1)
        expected = Series(np.zeros(N, dtype=np.float),
                          index=np.arange(N, dtype=np.float)).iloc[1:]
        expected.index.name = 'lagt'
        expected.name = 'msd'
        # HACK: Float64Index imprecision ruins index equality.
        # Test them separately. If that works, make them exactly the same.
        assert_almost_equal(actual.index.values, expected.index.values)
        actual.index = expected.index
        assert_series_equal(actual, expected)

    def test_linear_emsd(self):
        A = 1
        EARLY = 7 # only early lag times have good stats
        actual = tp.emsd(self.many_walks, 1, 1, max_lagtime=EARLY)
        a = np.arange(EARLY+1, dtype='float64')
        expected = Series(2*A*a, index=a).iloc[1:]
        expected.name = 'msd'
        expected.index.name = 'lagt'
        # HACK: Float64Index imprecision ruins index equality.
        # Test them separately. If that works, make them exactly the same.
        assert_almost_equal(actual.index.values, expected.index.values)
        actual.index = expected.index
        assert_series_equal(np.round(actual), expected)

    def test_linear_emsd_gaps(self):
        A = 1
        EARLY = 4  # only early lag times have good stats
        gapped_walks = self.many_walks.reset_index(drop=True)
        to_drop = np.random.choice(gapped_walks.index,
                                   int(len(gapped_walks) * 0.1), replace=False)
        gapped_walks = gapped_walks.drop(to_drop, axis=0)

        actual = tp.emsd(gapped_walks, 1, 1, max_lagtime=EARLY)
        a = np.arange(EARLY+1, dtype='float64')
        expected = Series(2*A*a, index=a).iloc[1:]
        expected.name = 'msd'
        expected.index.name = 'lagt'
        # HACK: Float64Index imprecision ruins index equality.
        # Test them separately. If that works, make them exactly the same.
        assert_almost_equal(actual.index.values, expected.index.values)
        actual.index = expected.index
        assert_series_equal(np.round(actual), expected)


class TestSpecial(StrictTestCase):
    def setUp(self):
        N = 10
        Y = 1
        a = DataFrame({'x': np.arange(N), 'y': np.zeros(N),
                      'frame': np.arange(N), 'particle': np.zeros(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.zeros(N - 1),
                       'frame': np.arange(1, N), 'particle': np.ones(N - 1)})
        self.steppers = conformity(pd.concat([a, b]))

    def test_theta_entropy(self):
        # just a smoke test
        theta_entropy = lambda x: tp.motion.theta_entropy(x, plot=False)
        self.steppers.groupby('particle').apply(theta_entropy)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
