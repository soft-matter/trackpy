import os
from copy import copy
import unittest

import numpy as np
import pandas as pd
from pandas import DataFrame
from numpy.testing import assert_equal

from trackpy.try_numba import NUMBA_AVAILABLE
from trackpy.utils import pandas_sort, pandas_concat
from trackpy.linking import (link, link_iter, link_df_iter, verify_integrity,
                             SubnetOversizeException, Linker, link_partial)
from trackpy.linking.subnetlinker import subnet_linker_recursive
from trackpy.tests.common import assert_traj_equal, StrictTestCase

path, _ = os.path.split(os.path.abspath(__file__))
path = os.path.join(path, 'data')


def random_walk(N):
    return np.cumsum(np.random.randn(N))



def _skip_if_no_numba():
    if not NUMBA_AVAILABLE:
        raise unittest.SkipTest('numba not installed. Skipping.')


SKLEARN_AVAILABLE = True
try:
    from sklearn.neighbors import BallTree
except ImportError:
    SKLEARN_AVAILABLE = False


def _skip_if_no_sklearn():
    if not SKLEARN_AVAILABLE:
        raise unittest.SkipTest('Scikit-learn not installed. Skipping.')

def unit_steps():
    return pd.DataFrame(dict(x=np.arange(5), y=5, frame=np.arange(5)))

random_x = np.random.randn(5).cumsum()
random_x -= random_x.min()  # All x > 0
max_disp = np.abs(np.diff(random_x)).max()


def random_walk_legacy():
    return pd.DataFrame(dict(x=random_x, y=0, frame=np.arange(5)))


def contracting_grid():
    """Two frames with a grid of 441 points.

    In the second frame, the points contract, so that the outermost set
    coincides with the second-outermost set in the previous frame.

    This is a way to challenge (and/or stump) a subnet solver.
    """
    pts0x, pts0y = np.mgrid[-10:11, -10:11] * 2.
    pts0 = pd.DataFrame(dict(x=pts0x.flatten(), y=pts0y.flatten(),
                             frame=0))
    pts1 = pts0.copy()
    pts1.frame = 1
    pts1.x = pts1.x * 0.9
    pts1.y = pts1.y * 0.9
    allpts = pandas_concat([pts0, pts1], ignore_index=True)
    allpts.x += 200  # Because BTree doesn't allow negative coordinates
    allpts.y += 200
    return allpts


class CommonTrackingTests(StrictTestCase):
    def setUp(self):
        self.linker_opts = dict(link_strategy='recursive')

    def test_one_trivial_stepper(self):
        # One 1D stepper
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        expected = f.copy()
        expected['particle'] = np.zeros(N)
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)

    def test_output_dtypes(self):
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N),
                       'frame': np.arange(N)})
        # Integer-typed input
        f['frame'] = f['frame'].astype(int)
        actual = self.link(f, 5)

        # Particle and frame columns should be integer typed
        assert np.issubdtype(actual['particle'], np.integer)
        assert np.issubdtype(actual['frame'], np.integer)

        # Float-typed input
        f['frame'] = f['frame'].astype(float)
        actual = self.link(f, 5)

        # Particle and frame columns should be integer typed
        assert np.issubdtype(actual['particle'], np.integer)
        assert np.issubdtype(actual['frame'], np.integer)

    def test_two_isolated_steppers(self):
        N = 5
        Y = 25
        # Begin second feature one frame later than the first, so the particle
        # labeling (0, 1) is established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1), 'frame': np.arange(1, N)})
        f = pandas_concat([a, b])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([np.zeros(N), np.ones(N - 1)])
        pandas_sort(expected, ['particle', 'frame'], inplace=True)
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)

        # Sort rows by frame (normal use)
        actual = self.link(pandas_sort(f, 'frame'), 5)
        assert_traj_equal(actual, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link(f1, 5)
        assert_traj_equal(actual, expected)

    def test_two_isolated_steppers_one_gapped(self):
        N = 5
        Y = 25
        # Begin second feature one frame later than the first,
        # so the particle labeling (0, 1) is established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N),
                      'frame': np.arange(N)})
        a = a.drop(3).reset_index(drop=True)
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1),
                      'frame': np.arange(1, N)})
        f = pandas_concat([a, b])
        expected = f.copy()
        expected['particle'] = np.concatenate([np.array([0, 0, 0, 2]), np.ones(N - 1)])
        pandas_sort(expected, ['particle', 'frame'], inplace=True)
        expected.reset_index(drop=True, inplace=True)
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)
        # link_df_iter() tests not performed, because hash_size is
        # not knowable from the first frame alone.

        # Sort rows by frame (normal use)
        actual = self.link(pandas_sort(f, 'frame'), 5)
        assert_traj_equal(actual, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link(f1, 5)
        assert_traj_equal(actual, expected)

    def test_isolated_continuous_random_walks(self):
        # Two 2D random walks
        np.random.seed(0)
        N = 30
        Y = 250
        M = 20 # margin, because negative values raise OutOfHash
        a = DataFrame({'x': M + random_walk(N), 'y': M + random_walk(N), 'frame': np.arange(N)})
        b = DataFrame({'x': M + random_walk(N - 1), 'y': M + Y + random_walk(N - 1), 'frame': np.arange(1, N)})
        f = pandas_concat([a, b])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([np.zeros(N), np.ones(N - 1)])
        pandas_sort(expected, ['particle', 'frame'], inplace=True)
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)

        # Many 2D random walks
        np.random.seed(0)
        initial_positions = [(100, 100), (200, 100), (100, 200), (200, 200)]
        import itertools
        c = itertools.count()
        def walk(x, y):
            i = next(c)
            return DataFrame({'x': x + random_walk(N - i),
                              'y': y + random_walk(N - i),
                             'frame': np.arange(i, N)})
        f = pandas_concat([walk(*pos) for pos in initial_positions])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([i*np.ones(N - i) for i in range(len(initial_positions))])
        pandas_sort(expected, ['particle', 'frame'], inplace=True)
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)

    def test_start_at_frame_other_than_zero(self):
        # One 1D stepper
        N = 5
        FIRST_FRAME = 3
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N),
                      'frame': FIRST_FRAME + np.arange(N)})
        expected = f.copy()
        expected['particle'] = np.zeros(N)
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)

    def test_blank_frame_no_memory(self):
        # One 1D stepper
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N),
                      'frame': [0, 1, 2, 4, 5],
                      'particle': [0, 0, 0, 1, 1]})
        expected = f.copy()
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)

    def test_real_data_that_causes_duplicate_bug(self):
        filename = 'reproduce_duplicate_track_assignment.npy'
        f = pd.DataFrame(np.load(os.path.join(path, filename)))
        # Not all parameters reproduce it, but these do
        f = self.link(f, 8, memory=2)
        verify_integrity(f)

    def test_search_range(self):
        t = self.link(unit_steps(), 1.1)
        assert_equal(t['particle'].values, 0)  # One track

        t_short = self.link(unit_steps(), 0.9)
        # Each step is a separate track.
        assert len(np.unique(t_short['particle'].values)) == len(t_short)

        t = self.link(random_walk_legacy(), max_disp + 0.1)
        assert_equal(t['particle'].values, 0)  # One track
        t_short = self.link(random_walk_legacy(), max_disp - 0.1)
        assert len(np.unique(t_short['particle'].values)) > 1  # Multiple tracks

    def test_easy_tracking(self):
        level_count = 5
        p_count = 16
        levels = []

        for j in range(level_count):
            for k in np.arange(p_count) * 2:
                levels.append([j, j, k])

        f = pd.DataFrame(levels, columns=['frame', 'x', 'y'])
        linked = self.link(f, 1.5)

        assert len(np.unique(linked['particle'].values)) == p_count

        for _, t in linked.groupby('particle'):
            x, y = t[['x', 'y']].values.T
            dx = np.diff(x)
            dy = np.diff(y)

            assert np.sum(dx) == level_count - 1
            assert np.sum(dy) == 0

    def test_copy(self):
        """Check inplace/copy behavior of link_df """
        # One 1D stepper
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        expected = f.copy()
        expected['particle'] = np.zeros(N)

        # Should copy
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)
        assert 'particle' not in f.columns

    def test_custom_to_eucl(self):
        # Several 2D random walkers
        N = 5
        length = 5
        step_size = 2
        search_range = 3

        steps = (np.random.random((2, length, N)) - 0.5) * step_size
        x, y = np.cumsum(steps, axis=2)
        f = DataFrame(dict(x=x.ravel(), y=y.ravel(),
                           frame=np.repeat(np.arange(length), N)))

        # link in normal (2D Euclidean) coordinates
        expected = self.link(f, search_range)

        # compute radial coordinates
        f_radial = f.copy()
        f_radial['angle'] = np.arctan2(f_radial['y'], f_radial['x'])
        f_radial['r'] = np.sqrt(f_radial['y'] ** 2 + f_radial['x'] ** 2)
        # leave x, y for the comparison at the end

        def to_eucl(arr):
            r, angle = arr.T
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            return np.array([x, y]).T

        # link using a custom distance function
        actual = self.link(f_radial, search_range, pos_columns=['r', 'angle'],
                           to_eucl=to_eucl)
        assert_traj_equal(actual, expected)

    def test_oversize_fail(self):
        with self.assertRaises(SubnetOversizeException):
            df = contracting_grid()
            self.link(df, search_range=2)

    def test_adaptive_fail(self):
        """Check recursion limit"""
        with self.assertRaises(SubnetOversizeException):
            self.link(contracting_grid(), search_range=2, adaptive_stop=1.84)

    def link(self, f, search_range, *args, **kwargs):
        kwargs = dict(self.linker_opts, **kwargs)
        return link(f, search_range, *args, **kwargs)


class SubnetNeededTests(CommonTrackingTests):
    def test_two_nearby_steppers(self):
        N = 5
        Y = 2
        # Begin second feature one frame later than the first, so the particle
        # labeling (0, 1) is established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1), 'frame': np.arange(1, N)})
        f = pandas_concat([a, b])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([np.zeros(N), np.ones(N - 1)])
        pandas_sort(expected, ['particle', 'frame'], inplace=True)
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)

        # Sort rows by frame (normal use)
        actual = self.link(pandas_sort(f, 'frame'), 5)
        assert_traj_equal(actual, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link(f1, 5)
        assert_traj_equal(actual, expected)

    def test_two_nearby_steppers_one_gapped(self):
        N = 5
        Y = 2
        # Begin second feature one frame later than the first, so the particle
        # labeling (0, 1) is established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1), 'frame': np.arange(1, N)})
        a = a.drop(3).reset_index(drop=True)
        f = pandas_concat([a, b])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([np.array([0, 0, 0, 2]), np.ones(N - 1)])
        pandas_sort(expected, ['particle', 'frame'], inplace=True)
        expected.reset_index(drop=True, inplace=True)
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)

        # Sort rows by frame (normal use)
        actual = self.link(pandas_sort(f, 'frame'), 5)
        assert_traj_equal(actual, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link(f1, 5)
        assert_traj_equal(actual, expected)

    def test_nearby_continuous_random_walks(self):
        # Two 2D random walks
        np.random.seed(0)
        N = 30
        Y = 250
        M = 20 # margin, because negative values raise OutOfHash
        a = DataFrame({'x': M + random_walk(N),
                       'y': M + random_walk(N),
                       'frame': np.arange(N)})
        b = DataFrame({'x': M + random_walk(N - 1),
                       'y': M + Y + random_walk(N - 1),
                       'frame': np.arange(1, N)})
        f = pandas_concat([a, b])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([np.zeros(N), np.ones(N - 1)])
        pandas_sort(expected, ['particle', 'frame'], inplace=True)
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)

        np.random.seed(0)
        initial_positions = [(10, 11), (10, 18), (14, 15), (20, 21), (13, 13),
                             (10, 10), (17, 19)]
        import itertools
        c = itertools.count()
        def walk(x, y):
            i = next(c)
            return DataFrame({'x': x + random_walk(N - i),
                              'y': y + random_walk(N - i),
                              'frame': np.arange(i, N)})
        f = pandas_concat([walk(*pos) for pos in initial_positions])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([i*np.ones(N - i) for i in range(len(initial_positions))])
        pandas_sort(expected, ['particle', 'frame'], inplace=True)
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link(f1, 5)
        assert_traj_equal(actual, expected)

    def test_quadrature_distances(self):
        """A simple test to check whether the subnet linker adds
        orthogonal coordinates in quadrature (as in Pythagoras).

        We have two possible linking results:

        1. A->C and B->D, cost (linear) = 16, cost (quadrature) = 200
        2. A->D and B->C, cost (linear) = 28, cost (quadrature) = 200
        """
        def subnet_test(epsilon):
            """Returns 2 features in 2 frames, which represent a special
            case when the subnet linker adds distances in quadrature. With
            epsilon=0, subnet linking is degenerate. Therefore
            linking should differ for positive and negative epsilon."""
            return pd.DataFrame([(0, 6, 0),             #A
                                 (0, 14 + epsilon, 8),  #B
                                 (1, 8, 0),             #C
                                 (1, 0, 8)],            #D
                                columns=['frame', 'x', 'y'])

        trpos = self.link(subnet_test(1), 20)
        expected = subnet_test(1)
        expected['particle'] = np.array([0, 1, 1, 0])
        assert_traj_equal(trpos, expected)

        trneg = self.link(subnet_test(-1), 20)
        expected = subnet_test(-1)
        expected['particle'] = np.array([0, 1, 0, 1])
        assert_traj_equal(trneg, expected)

    def test_quadrature_sum(self):
        """A simple test to check whether the subnet linker adds
        distances in quadrature (as in Crocker-Grier)."""
        def subnet_test(epsilon):
            """Returns 2 features in 2 frames, which represent a special
            case when the subnet linker adds distances in quadrature. With
            epsilon=0, subnet linking is degenerate. Therefore
            linking should differ for positive and negative epsilon."""
            return pd.DataFrame([(0, 10, 30), (0, 10, 0),
                                 (1, 0, 20), (1, 30, 20 + epsilon)],
                         columns=['frame', 'x', 'y'])

        trpos = self.link(subnet_test(1), 30)
        expected = subnet_test(1)
        expected['particle'] = np.array([0, 1, 1, 0])
        assert_traj_equal(trpos, expected)

        trneg = self.link(subnet_test(-1), 30)
        expected = subnet_test(-1)
        expected['particle'] = np.array([0, 1, 0, 1])
        assert_traj_equal(trneg, expected)

    def test_penalty(self):
        """A test case of two particles, spaced 8 and each moving by 8 down
        and 7 to the right. We have two likely linking results:

        1. two links, total squared displacement = 2*(8**2 + 7**2) = 226
        2. one link, total squared displacement = (8**2 + 1**2) + sr**2

        Case 2 gets a penalty for not linking, which equals the search range
        squared. We vary this in this test.

        With a penalty of 13, case 2 has a total cost of 234 and we expect case
        1. as the result.

        With a penalty of 12, case 2. will have a total cost of 209 and we
        expect case 2. as the result.
        """
        f = pd.DataFrame({'x': [0, 8, 7, 8 + 7],
                          'y': [0, 0, 8, 8],
                          'frame': [0, 0, 1, 1]})
        case1 = f.copy()
        case1['particle'] = np.array([0, 1, 0, 1])
        case2 = f.copy()
        case2['particle'] = np.array([0, 1, 1, 2])

        actual = self.link(f, 13)
        pandas_sort(case1, ['x'], inplace=True)
        pandas_sort(actual, ['x'], inplace=True)
        assert_equal(actual['particle'].values.astype(int),
                     case1['particle'].values.astype(int))

        actual = self.link(f, 12)
        pandas_sort(case2, ['x'], inplace=True)
        pandas_sort(actual, ['x'], inplace=True)
        assert_equal(actual['particle'].values.astype(int),
                     case2['particle'].values.astype(int))

    def test_memory(self):
        """A unit-stepping trajectory and a random walk are observed
        simultaneously. The random walk is missing from one observation."""
        a = unit_steps()
        b = random_walk_legacy()
        # b[2] is intentionally omitted below.
        gapped = pandas_concat([a, b[b['frame'] != 2]])

        safe_disp = 1 + random_x.max() - random_x.min()  # Definitely large enough
        t0 = self.link(gapped, safe_disp, memory=0)
        assert len(np.unique(t0['particle'].values)) == 3
        t2 = self.link(gapped, safe_disp, memory=2)
        assert len(np.unique(t2['particle'].values)) == 2

    def test_memory_removal(self):
        """BUG: A particle remains in memory after its Track is resumed, leaving two
        copies that can independently pick up desinations, leaving two Points in the
        same Track in a single level."""
        levels  = []
        levels.extend([[0, 1, 1], [0, 4, 1]])  # two points
        levels.extend([[1, 1, 1]])  # one vanishes, but is remembered
        levels.extend([[2, 1, 1], [2, 2, 1]])  # resume Track
        levels.extend([[3, 1, 1], [3, 2, 1], [3, 4, 1]])
        f = pd.DataFrame(levels, columns=['frame', 'x', 'y'])
        t = self.link(f, 5, memory=2)
        assert len(np.unique(t['particle'].values)) == 3

    def test_memory_with_late_appearance(self):
        a = unit_steps()
        b = random_walk_legacy()
        gapped = pandas_concat([a, b[b['frame'].isin([1, 4])]])

        safe_disp = 1 + random_x.max() - random_x.min()  # large enough
        t0 = self.link(gapped, safe_disp, memory=1)
        assert len(np.unique(t0['particle'].values)) == 3
        t2 = self.link(gapped, safe_disp, memory=4)
        assert len(np.unique(t2['particle'].values)) == 2

    def test_memory_on_one_gap(self):
        N = 5
        Y = 2
        # Begin second feature one frame later than the first, so the particle labeling (0, 1) is
        # established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1), 'frame': np.arange(1, N)})
        a = a.drop(3).reset_index(drop=True)
        f = pandas_concat([a, b])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([np.array([0, 0, 0, 0]), np.ones(N - 1)])
        pandas_sort(expected, ['particle', 'frame'], inplace=True)
        expected.reset_index(drop=True, inplace=True)
        actual = self.link(f, 5, memory=1)
        assert_traj_equal(actual, expected)

        # Sort rows by frame (normal use)
        actual = self.link(pandas_sort(f, 'frame'), 5, memory=1)
        assert_traj_equal(actual, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link(f1, 5, memory=1)
        assert_traj_equal(actual, expected)

    def test_pathological_tracking(self):
        level_count = 5
        p_count = 16
        levels = []
        shift = 1

        for j in range(level_count):
            for k in np.arange(p_count) * 2:
                levels.append([j, j, k + j * shift])

        f = pd.DataFrame(levels, columns=['frame', 'x', 'y'])
        linked = self.link(f, 8)

        assert len(np.unique(linked['particle'].values)) == p_count

    def test_adaptive_range(self):
        """Tests that is unbearably slow without a fast subnet linker."""
        cg = contracting_grid()
        # Allow 5 applications of the step
        tracks = self.link(cg, 2, adaptive_step=0.8, adaptive_stop=0.64)
        # Transform back to origin
        tracks.x -= 200
        tracks.y -= 200
        assert len(cg) == len(tracks)
        tr0 = tracks[tracks.frame == 0].set_index('particle')
        tr1 = tracks[tracks.frame == 1].set_index('particle')
        only0 = list(set(tr0.index) - set(tr1.index))
        only1 = list(set(tr1.index) - set(tr0.index))
        # From the first frame, the outermost particles should have been lost.
        assert all(
            (tr0.x.loc[only0].abs() > 19) | (tr0.y.loc[only0].abs() > 19))
        # There should be new tracks in the second frame, corresponding to the
        # middle radii.
        assert all(
            (tr1.x.loc[only1].abs() == 9) | (tr1.y.loc[only1].abs() == 9))


class SimpleLinkingTestsIter(CommonTrackingTests):
    def link(self, f, search_range, *args, **kwargs):
        pos_columns = kwargs.pop('pos_columns', ['y', 'x'])

        def f_iter(f, first_frame, last_frame):
            """ link_iter requires an (optionally enumerated) generator of
            ndarrays """
            for t in np.arange(first_frame, last_frame + 1,
                               dtype=f['frame'].dtype):
                f_filt = f[f['frame'] == t]
                yield t, f_filt[pos_columns].values

        res = f.copy()
        res['particle'] = -1
        for t, ids in link_iter(f_iter(f, 0, int(f['frame'].max())),
                                search_range, *args, **kwargs):
            res.loc[res['frame'] == t, 'particle'] = ids
        return pandas_sort(res, ['particle', 'frame']).reset_index(drop=True)

    def test_output_dtypes(self):
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N),
                       'frame': np.arange(N)})
        # Integer-typed input
        f['frame'] = f['frame'].astype(int)
        actual = self.link(f, 5)

        # Particle and frame columns should be integer typed
        assert np.issubdtype(actual['particle'], np.integer)
        assert np.issubdtype(actual['frame'], np.integer)

        # Float-typed input: frame column type is propagated in link_iter
        f['frame'] = f['frame'].astype(float)
        actual = self.link(f, 5)
        assert np.issubdtype(actual['particle'], np.integer)
        assert np.issubdtype(actual['frame'], np.floating)


class SimpleLinkingTestsDfIter(CommonTrackingTests):
    def link(self, f, search_range, *args, **kwargs):

        def df_iter(f, first_frame, last_frame):
            """ link_df_iter requires a generator of dataframes """
            for t in range(first_frame, last_frame + 1):
                yield f[f['frame'] == t]

        res_iter = link_df_iter(df_iter(f, 0, int(f['frame'].max())),
                                search_range, *args, **kwargs)
        res = pandas_concat(res_iter)
        return pandas_sort(res, ['particle', 'frame']).reset_index(drop=True)

    def test_output_dtypes(self):
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N),
                       'frame': np.arange(N)})
        # Integer-typed input
        f['frame'] = f['frame'].astype(int)
        actual = self.link(f, 5)

        # Particle and frame columns should be integer typed
        assert np.issubdtype(actual['particle'], np.integer)
        assert np.issubdtype(actual['frame'], np.integer)

        # Float-typed input: frame column type is propagated in link_df_iter
        f['frame'] = f['frame'].astype(float)
        actual = self.link(f, 5)
        assert np.issubdtype(actual['particle'], np.integer)
        assert np.issubdtype(actual['frame'], np.floating)


class TestDropLink(CommonTrackingTests):
    def setUp(self):
        self.linker_opts = dict(link_strategy='drop')

    def test_drop_link(self):
        # One 1D stepper. A new particle appears in frame 2.
        # The resulting subnet causes the trajectory to be broken
        # when link_strategy is 'drop' and search_range is large enough.
        N = 2
        f_1particle = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        f = f_1particle.append(DataFrame(
            {'x': [3], 'y': [1], 'frame': [1]}), ignore_index=True)
        f_expected_without_subnet = f.copy()
        f_expected_without_subnet['particle'] = [0, 0, 1]
        # The linker assigns new particle IDs in arbitrary order. So
        # comparing with expected values is tricky.
        # We just check for the creation of 2 new trajectories.
        without_subnet = self.link(f, 1.5)
        assert_traj_equal(without_subnet, f_expected_without_subnet)
        with_subnet = self.link(f, 5)
        assert set(with_subnet.particle) == {0, 1, 2}


class TestNumbaLink(SubnetNeededTests):
    def setUp(self):
        _skip_if_no_numba()
        self.linker_opts = dict(link_strategy='numba')


class TestHybridLink(SubnetNeededTests):
    def setUp(self):
        _skip_if_no_numba()
        self.linker_opts = dict(link_strategy='hybrid')


class TestNonrecursiveLink(SubnetNeededTests):
    def setUp(self):
        self.linker_opts = dict(link_strategy='nonrecursive')


class TestBTreeLink(SubnetNeededTests):
    def setUp(self):
        _skip_if_no_sklearn()
        self.linker_opts = dict(neighbor_strategy='BTree')

    def test_custom_dist_pyfunc(self):
        # Several 2D random walkers
        N = 5
        length = 5
        step_size = 2
        search_range = 3

        steps = (np.random.random((2, length, N)) - 0.5) * step_size
        x, y = np.cumsum(steps, axis=2)
        f = DataFrame(dict(x=x.ravel(), y=y.ravel(),
                           frame=np.repeat(np.arange(length), N)))

        # link in normal (2D Euclidean) coordinates
        expected = self.link(f, search_range)

        # compute radial coordinates
        f_radial = f.copy()
        f_radial['angle'] = np.arctan2(f_radial['y'], f_radial['x'])
        f_radial['r'] = np.sqrt(f_radial['y']**2 + f_radial['x']**2)
        # leave x, y for the comparison at the end

        def dist_func(a, b):
            x1 = a[0] * np.cos(a[1])
            y1 = a[0] * np.sin(a[1])
            x2 = b[0] * np.cos(b[1])
            y2 = b[0] * np.sin(b[1])

            return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

        # link using a custom distance function
        actual = self.link(f_radial, search_range, pos_columns=['r', 'angle'],
                           dist_func=dist_func)
        assert_traj_equal(actual, expected)


    def test_custom_dist_metric(self):
        import sklearn.neighbors

        # Several 2D random walkers
        N = 5
        length = 5
        step_size = 2
        search_range = 3

        steps = (np.random.random((2, length, N)) - 0.5) * step_size
        x, y = np.cumsum(steps, axis=2)
        f = DataFrame(dict(x=x.ravel(), y=y.ravel(),
                           frame=np.repeat(np.arange(length), N)))

        # link in normal (2D Euclidean) coordinates
        expected = self.link(f, search_range)

        # leave x, y for the comparison at the end

        dist_func = sklearn.neighbors.DistanceMetric.get_metric("euclidean")
        
        # link using a custom distance function
        actual = self.link(f, search_range, dist_func=dist_func)
        assert_traj_equal(actual, expected)


class TestMockSubnetlinker(StrictTestCase):
    def setUp(self):
        self.dest = []
        self.source = []
        self.sr = []

        def copy_point(point):
            return dict(id=point.id, pos=point.pos, t=point.t,
                        forward_cands=copy(point.forward_cands))

        def mock_subnetlinker(source_set, dest_set, search_range, **kwargs):
            self.source.append([copy_point(p) for p in source_set])
            self.dest.append([copy_point(p) for p in dest_set])
            self.sr.append(search_range)

            return subnet_linker_recursive(source_set, dest_set, search_range,
                                           **kwargs)

        self.linker_opts = dict(link_strategy=mock_subnetlinker)
        self.default_max_size = Linker.MAX_SUB_NET_SIZE_ADAPTIVE

    def tearDown(self):
        Linker.MAX_SUB_NET_SIZE_ADAPTIVE = self.default_max_size

    def link(self, *args, **kwargs):
        kwargs.update(self.linker_opts)
        return link(*args, **kwargs)

    def test_single_subnet(self):
        f = DataFrame({'x': [0, 1, 2, 3, 4],
                       'frame': [0, 1, 1, 1, 1]})
        search_range = 10.

        self.link(f, search_range=search_range, pos_columns=['x'])

        source = self.source[0]
        dest = self.dest[0]
        sr = self.sr[0]

        self.assertEqual(sr, search_range)
        self.assertEqual(len(source), 1)
        self.assertEqual(len(dest), 4)

        fwd_cds = source[0]['forward_cands']

        # there are no forward candidates inside search_range
        for p, dist in fwd_cds:
            self.assertLessEqual(dist, search_range)

        # the forward candidate distances are sorted in ascending order
        for i in range(len(fwd_cds) - 1):
            self.assertLess(fwd_cds[i][1], fwd_cds[i + 1][1])

    def test_adaptive_subnet(self):
        Linker.MAX_SUB_NET_SIZE_ADAPTIVE = 1

        f = DataFrame({'x': [0.0, 1.0, 0.1, 1.1],
                       'frame': [0, 0, 1, 1]})
        search_range = 8
        self.link(f, search_range=search_range, pos_columns=['x'],
                  adaptive_step=0.25, adaptive_stop=0.25)

        # round 0: search range 8, one call to subnetlinker
        self.assertEqual(self.sr[0], 8.)
        self.assertEqual(len(self.source[0]), 2)
        self.assertEqual(len(self.dest[0]), 2)
        for p in self.source[0]:
            self.assertEqual(len(p['forward_cands']), 2)
            for cand, dist in p['forward_cands']:
                self.assertIsNot(cand, None)
                self.assertLess(dist, self.sr[0])

        # round 1: search range 2; same
        self.assertEqual(self.sr[1], 2.)
        self.assertEqual(len(self.source[1]), 2)
        self.assertEqual(len(self.dest[1]), 2)
        for p in self.source[1]:
            for cand, dist in p['forward_cands']:
                self.assertIsNot(cand, None)
                self.assertLess(dist, self.sr[1])

        # round 2, both calls: search range 0.5, subnets separate
        for i in (2, 3):
            self.assertEqual(self.sr[i], 0.5)
            self.assertEqual(len(self.source[i]), 1)
            self.assertEqual(len(self.dest[i]), 1)
            for p in self.source[i]:
                self.assertEqual(len(p['forward_cands']), 1)
                self.assertAlmostEqual(p['forward_cands'][0][1], 0.1)


class TestPartialLink(CommonTrackingTests):
    def test_patch_single(self):
        f = DataFrame({'x':        [0, 1, 2, 3],
                       'y':        [1, 1, 1, 1],
                       'frame':    [0, 1, 2, 3],
                       'particle': [3, 3, 4, 4]})

        actual = link_partial(f, 5, link_range=(1, 3))
        assert_equal(actual['particle'].values, 3)

    def test_patch_crossing(self):
        f = DataFrame({'x':        [0, 1, 2, 3, 0, 1, 2, 3],
                       'y':        [1, 1, 1, 1, 5, 5, 5, 5],
                       'frame':    [0, 1, 2, 3, 0, 1, 2, 3],
                       'particle': [3, 3, 4, 4, 4, 4, 3, 3]})

        actual = link_partial(f, 5, link_range=(1, 3))
        assert_equal(actual.loc[actual['y'] == 1, 'particle'].values[:4], 3)
        assert_equal(actual.loc[actual['y'] == 5, 'particle'].values[4:], 4)

    def test_patch_appearing(self):
        f = DataFrame({'x':        [0, 1, 2, 3, 2, 3],
                       'y':        [1, 1, 1, 1, 5, 5],
                       'frame':    [0, 1, 2, 3, 2, 3],
                       'particle': [3, 3, 4, 4, 3, 3]})

        actual = link_partial(f, 5, link_range=(1, 3))
        assert_equal(actual.loc[actual['y'] == 1, 'particle'].values[:4], 3)
        assert_equal(actual.loc[actual['y'] == 5, 'particle'].values[4:], 4)

    def test_patch_only_in(self):
        f = DataFrame({'x':        [0, 1, 2, 3, 2, 3],
                       'y':        [1, 1, 1, 1, 5, 5],
                       'frame':    [0, 1, 2, 3, 1, 2],
                       'particle': [3, 3, 4, 4, 3, 3]})

        actual = link_partial(f, 5, link_range=(1, 3))
        assert_equal(actual.loc[actual['y'] == 1, 'particle'].values[:4], 3)
        assert_equal(actual.loc[actual['y'] == 5, 'particle'].values[4:], 0)
