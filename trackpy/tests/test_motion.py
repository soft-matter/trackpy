import numpy as np
from pandas import DataFrame, Series

from pandas.testing import (
    assert_series_equal,
    assert_frame_equal,
)

from numpy.testing import assert_almost_equal

import trackpy as tp
from trackpy.utils import pandas_sort, pandas_concat, is_pandas_since_220
from trackpy.tests.common import StrictTestCase


def random_walk(N):
    return np.cumsum(np.random.randn(N))


def conformity(df):
    """ Organize toy data to look like real data. Be strict about dtypes:
    particle is a float and frame is an integer."""
    df['frame'] = df['frame'].astype(np.int64)
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    df.set_index('frame', drop=False, inplace=True)
    if 'particle' in df.columns:
        df['particle'] = df['particle'].astype(float)
        return pandas_sort(df, by=['frame', 'particle'])
    else:
        return pandas_sort(df, by=['frame'])


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
        self.dead_still = conformity(pandas_concat([a, b]))
        pandas_sort(self.dead_still, ['frame', 'particle'], inplace=True)

        P = 1000 # particles
        A = 0.00001 # step amplitude
        np.random.seed(0)
        particles = [DataFrame({'x': A*random_walk(N), 'y': A*random_walk(N),
                                'frame': np.arange(N), 'particle': i})
                     for i in range(P)]
        self.many_walks = conformity(pandas_concat(particles))

        self.unlabeled_walks = self.many_walks.copy()
        del self.unlabeled_walks['particle']

        a = DataFrame({'x': np.arange(N), 'y': np.zeros(N),
                       'frame': np.arange(N), 'particle': np.zeros(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.zeros(N - 1),
                       'frame': np.arange(1, N), 'particle': np.ones(N - 1)})
        self.steppers = conformity(pandas_concat([a, b]))

        # Single-particle trajectory with no particle label
        self.single_stepper = conformity(a.copy())
        del self.single_stepper['particle']

    def test_no_drift(self):
        N = 10
        expected = DataFrame({'x': np.zeros(N), 'y': np.zeros(N)}).iloc[1:]
        expected = expected.astype('float')
        expected.index.name = 'frame'
        expected.columns = ['x', 'y']
        # ^ no drift measured for Frame 0

        actual = tp.compute_drift(self.dead_still)
        assert_frame_equal(actual, expected[['y', 'x']])

        actual_rolling = tp.compute_drift(self.dead_still, smoothing=2)
        assert_frame_equal(actual_rolling, expected[['y', 'x']])

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
                          np.arange(1, N, dtype=int)).astype('float64')
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
                          index=np.arange(1, N, dtype=int)).astype('float64')
        drift.columns = ['x', 'y']
        drift.index.name = 'frame'
        actual = tp.subtract_drift(add_drift(self.dead_still, drift), drift)
        assert_traj_equal(actual, self.dead_still)
        actual = tp.subtract_drift(add_drift(self.many_walks, drift), drift)
        assert_traj_equal(actual, self.many_walks)
        actual = tp.subtract_drift(add_drift(self.steppers, drift), drift)
        assert_traj_equal(actual, self.steppers)

        actual = tp.subtract_drift(add_drift(self.single_stepper, drift), drift)
        assert_traj_equal(actual, self.single_stepper)

        # Test that subtract_drift is OK without particle labels.
        # In principle, Series.sub() may raise an error because
        # the 'frame' index is duplicated.
        # Don't check the result since we can't compare unlabeled trajectories!
        actual = tp.subtract_drift(add_drift(self.unlabeled_walks, drift),
                                   drift)


class TestMSD(StrictTestCase):
    def setUp(self):
        N = 10
        Y = 1
        a = DataFrame({'x': np.zeros(N), 'y': np.zeros(N),
                      'frame': np.arange(N), 'particle': np.zeros(N)})
        b = DataFrame({'x': np.zeros(N - 1), 'y': Y + np.zeros(N - 1),
                       'frame': np.arange(1, N), 'particle': np.ones(N - 1)})
        self.dead_still = conformity(pandas_concat([a, b]))

        P = 50 # particles
        A = 1 # step amplitude
        np.random.seed(0)
        particles = [DataFrame({'x': A*random_walk(N), 'y': A*random_walk(N),
                                'frame': np.arange(N), 'particle': i})
                     for i in range(P)]
        self.many_walks = conformity(pandas_concat(particles))

        a = DataFrame({'x': np.arange(N), 'y': np.zeros(N),
                      'frame': np.arange(N), 'particle': np.zeros(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.zeros(N - 1),
                       'frame': np.arange(1, N), 'particle': np.ones(N - 1)})
        self.steppers = conformity(pandas_concat([a, b]))

        self.badly_gapped_walks = conformity(DataFrame({
            'frame': [1, 2, 3, 4, 5, 6, 1, 2, 6], # create an example trajectory where particle 2 disappears for a few frames and returns.
            'x': [4, 5, 4, 5, 3, 2, -3, -1, -2],
            'y': [6, 7, 8, 6, 7, 6, 10, 11, 10],
            'particle': [1, 1, 1, 1, 1, 1, 2, 2, 2]
        }))

    def test_zero_emsd(self):
        N = 10
        actual = tp.emsd(self.dead_still, 1, 1)
        expected = Series(np.zeros(N, dtype=float),
                          index=np.arange(N, dtype=float)).iloc[1:]
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

    def test_major_gap_imsd(self):
        """Large gap that should strongly affect emsd.
        
        Test data and fix (PR #773) by @vivarose
        """
        imsd = tp.imsd(self.badly_gapped_walks, mpp=1, fps=1)
        nans = imsd.isna()

        assert nans[2][2.0]
        assert nans[2][3.0]
        assert sum(nans.values.flatten()) == 2

    def test_major_gap_emsd(self):
        """Large gap that should strongly affect emsd.
        
        Test data and fix (PR #773) by @vivarose
        """
        actual = tp.emsd(self.badly_gapped_walks, mpp=1, fps=1)
        expected = Series({
            1.0: 4.012847555129437,
            2.0: 4.000000000000007,
            3.0: 4.333333333333333,
            4.0: 4.193672099712366,
            5.0: 2.645254074784259
            })
        assert_almost_equal(expected.values, actual.values)

    def test_direction_corr(self):
        # just a smoke test
        f1, f2 = 2, 6
        df = tp.motion.direction_corr(self.many_walks, f1, f2)
        P = len(self.many_walks.particle.unique())
        assert len(df) == (P * (P - 1)) / 2


class TestSpecial(StrictTestCase):
    def setUp(self):
        N = 10
        Y = 1
        a = DataFrame({'x': np.arange(N), 'y': np.zeros(N),
                      'frame': np.arange(N), 'particle': np.zeros(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.zeros(N - 1),
                       'frame': np.arange(1, N), 'particle': np.ones(N - 1)})
        self.steppers = conformity(pandas_concat([a, b]))

    def test_theta_entropy(self):
        # just a smoke test
        theta_entropy = lambda x: tp.motion.theta_entropy(x, plot=False)
        self.steppers.groupby('particle').apply(
            theta_entropy,
            **({"include_groups": False} if is_pandas_since_220 else {}),
        )

    def test_relate_frames(self):
        # Check completeness of output
        pos_columns = ['x', 'y']
        f1, f2 = 2, 6
        df = tp.motion.relate_frames(self.steppers, f1, f2, pos_columns=pos_columns)
        for c in pos_columns:
            assert c in df
            assert c + '_b' in df
            assert 'd' + c in df
        assert 'dr' in df
        assert 'direction' in df


if __name__ == '__main__':
    import unittest
    unittest.main()
