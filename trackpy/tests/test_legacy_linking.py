import os
from copy import deepcopy
import unittest
import warnings

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pandas import DataFrame

from trackpy.try_numba import NUMBA_AVAILABLE
from trackpy.linking.legacy import (link, link_df, link_df_iter,
                                    PointND, Hash_table, strip_diagnostics,
                                    SubnetOversizeException)
from trackpy.utils import pandas_sort, pandas_concat, validate_tuple
from trackpy.tests.common import StrictTestCase, assert_traj_equal


path, _ = os.path.split(os.path.abspath(__file__))
path = os.path.join(path, 'data')


# Call lambda function for a fresh copy each time.
unit_steps = lambda: [[PointND(t, (x, 0))] for t, x in enumerate(range(5))]

np.random.seed(0)
random_x = np.random.randn(5).cumsum()
random_x -= random_x.min()  # All x > 0
max_disp = np.diff(random_x).max()
random_walk_legacy = lambda: [[PointND(t, (x, 5))] 
                              for t, x in enumerate(random_x)]


def hash_generator(dims, box_size):
    return lambda: Hash_table(dims, box_size)


def _skip_if_no_numba():
    if not NUMBA_AVAILABLE:
        raise unittest.SkipTest('numba not installed. Skipping.')


def random_walk(N):
    return np.cumsum(np.random.randn(N))


def contracting_grid():
    """Two frames with a grid of 441 points.

    In the second frame, the points contract, so that the outermost set
    coincides with the second-outermost set in the previous frame.

    This is a way to challenge (and/or stump) a subnet solver.
    """
    pts0x, pts0y = np.mgrid[-10:11,-10:11]
    pts0 = pd.DataFrame(dict(x=pts0x.flatten(), y=pts0y.flatten(),
                             frame=0))
    pts1 = pts0.copy()
    pts1.frame = 1
    pts1.x = pts1.x * 0.9
    pts1.y = pts1.y * 0.9
    allpts = pandas_concat([pts0, pts1], ignore_index=True)
    allpts.x += 100  # Because BTree doesn't allow negative coordinates
    allpts.y += 100
    return allpts


class CommonTrackingTests:
    do_diagnostics = False  # Don't ask for diagnostic info from linker

    def test_one_trivial_stepper(self):
        # One 1D stepper
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        expected = f.copy()
        expected['particle'] = np.zeros(N)
        actual = self.link_df(f, 5)
        assert_traj_equal(actual, expected)
        actual_iter = self.link_df_iter(f, 5, hash_size=(10, 2))
        assert_traj_equal(actual_iter, expected)
        if self.do_diagnostics:
            assert 'diag_search_range' in self.diag.columns
            # Except for first frame, all particles should have been labeled
            # with a search_range
            assert not any(self.diag['diag_search_range'][
                               actual_iter.frame > 0].isnull())

    def test_two_isolated_steppers(self):
        N = 5
        Y = 25
        # Begin second feature one frame later than the first, so the particle labeling (0, 1) is
        # established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1), 'frame': np.arange(1, N)})
        f = pandas_concat([a, b])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([np.zeros(N), np.ones(N - 1)])
        pandas_sort(expected, ['particle', 'frame'], inplace=True)
        actual = self.link_df(f, 5)
        assert_traj_equal(actual, expected)
        actual_iter = self.link_df_iter(f, 5, hash_size=(50, 50))
        assert_traj_equal(actual_iter, expected)

        # Sort rows by frame (normal use)
        actual = self.link_df(pandas_sort(f, 'frame'), 5)
        assert_traj_equal(actual, expected)
        actual_iter = self.link_df_iter(pandas_sort(f, 'frame'), 5, hash_size=(50, 50))
        assert_traj_equal(actual_iter, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link_df(f1, 5)
        assert_traj_equal(actual, expected)
        actual_iter = self.link_df_iter(f1, 5, hash_size=(50, 50))
        assert_traj_equal(actual_iter, expected)

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
        actual = self.link_df(f, 5)
        assert_traj_equal(actual, expected)
        actual_iter = self.link_df_iter(f, 5, hash_size=(50, 50))
        assert_traj_equal(actual_iter, expected)
        # link_df_iter() tests not performed, because hash_size is
        # not knowable from the first frame alone.

        # Sort rows by frame (normal use)
        actual = self.link_df(pandas_sort(f, 'frame'), 5)
        assert_traj_equal(actual, expected)
        actual_iter = self.link_df_iter(pandas_sort(f, 'frame'), 5, hash_size=(50, 50))
        assert_traj_equal(actual_iter, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link_df(f1, 5)
        assert_traj_equal(actual, expected)
        actual_iter = self.link_df_iter(f1, 5, hash_size=(50, 50))
        assert_traj_equal(actual_iter, expected)

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
        actual = self.link_df(f, 5)
        assert_traj_equal(actual, expected)
        actual_iter = self.link_df_iter(f, 5, hash_size=(2*M, Y + 2*M))
        assert_traj_equal(actual_iter, expected)

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
        actual = self.link_df(f, 5)
        assert_traj_equal(actual, expected)
        actual_iter = self.link_df_iter(f, 5, hash_size=(200 + M, 200 + M))
        assert_traj_equal(actual_iter, expected)

    def test_start_at_frame_other_than_zero(self):
        # One 1D stepper
        N = 5
        FIRST_FRAME = 3
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N), 
                      'frame': FIRST_FRAME + np.arange(N)})
        expected = f.copy()
        expected['particle'] = np.zeros(N)
        actual = self.link_df(f, 5)
        assert_traj_equal(actual, expected)
        actual = self.link_df_iter(f, 5, hash_size=(6, 2))
        assert_traj_equal(actual, expected)

    def test_blank_frame_no_memory(self):
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N),
                      'frame': [0, 1, 2, 4, 5]})
        expected = f.copy()

        # Using link_df, the particle will be given a new ID after the gap.
        expected['particle'] = np.array([0, 0, 0, 1, 1], dtype=np.float64)
        actual = self.link_df(f, 5)
        assert_traj_equal(actual, expected)

        # link_df_iter will (in this test suite) iterate over only the frames
        # present in the dataframe, so the gap will be ignored.
        expected['particle'] = 0.0
        actual = self.link_df_iter(f, 5, hash_size=(10, 10))
        assert_traj_equal(actual, expected)


    def test_real_data_that_causes_duplicate_bug(self):
        filename = 'reproduce_duplicate_track_assignment.npy'
        f = pd.DataFrame(np.load(os.path.join(path, filename)))
        # Not all parameters reproduce it, but these do
        self.link_df(f, 8, 2, verify_integrity=True)


    def test_search_range(self):
        t = self.link(unit_steps(), 1.1, hash_generator((10, 10), 1))
        assert len(t) == 1  # One track
        t_short = self.link(unit_steps(), 0.9, hash_generator((10, 10), 1))
        assert len(t_short) == len(unit_steps())  # Each step is a separate track.

        t = self.link(random_walk_legacy(), max_disp + 0.1,
                 hash_generator((10, 10), 1))
        assert len(t) == 1  # One track
        t_short = self.link(random_walk_legacy(), max_disp - 0.1,
                       hash_generator((10, 10), 1))
        assert len(t_short) > 1  # Multiple tracks

    def test_box_size(self):
        """No matter what the box size, there should be one track, and it should
        contain all the points."""
        for box_size in [0.1, 1, 10]:
            t1 = self.link(unit_steps(), 1.1, hash_generator((10, 10), box_size))
            t2 = self.link(random_walk_legacy(), max_disp + 1,
                      hash_generator((10, 10), box_size))
            assert len(t1) == 1
            assert len(t2) == 1
            assert len(t1[0].points) == len(unit_steps())
            assert len(t2[0].points) == len(random_walk_legacy())
    
    def test_easy_tracking(self):
        level_count = 5
        p_count = 16
        levels = []

        for j in range(level_count):
            level = []
            for k in np.arange(p_count) * 2:
                level.append(PointND(j, (j, k)))
            levels.append(level)

        hash_generator = lambda: Hash_table((level_count + 1,
                                            p_count * 2 + 1), .5)
        tracks = self.link(levels, 1.5, hash_generator)
    
        assert len(tracks) == p_count
    
        for t in tracks:
            x, y = zip(*[p.pos for p in t])
            dx = np.diff(x)
            dy = np.diff(y)
    
            assert np.sum(dx) == level_count - 1
            assert np.sum(dy) == 0

    def test_copy(self):
        """Check inplace/copy behavior of link_df, link_df_iter"""
        # One 1D stepper
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        f_inplace = f.copy()
        expected = f.copy()
        expected['particle'] = np.zeros(N)

        # Should add particle column in-place
        # UNLESS diagnostics are enabled (or input dataframe is not writeable)
        actual = self.link_df(f_inplace, 5)
        assert_traj_equal(actual, expected)
        if self.do_diagnostics:
            assert 'particle' not in f_inplace.columns
        else:
            assert_traj_equal(actual, f_inplace)

        # Should copy
        actual = self.link_df(f, 5, copy_features=True)
        assert_traj_equal(actual, expected)
        assert 'particle' not in f.columns

        # Should copy
        actual_iter = self.link_df_iter(f, 5, hash_size=(10, 2))
        assert_traj_equal(actual_iter, expected)
        assert 'particle' not in f.columns

    def test_oversize_fail(self):
        with self.assertRaises(SubnetOversizeException):
            self.link_df(contracting_grid(), 1)

    def test_adaptive_fail(self):
        """Check recursion limit"""
        with self.assertRaises(SubnetOversizeException):
            self.link_df(contracting_grid(), 1, adaptive_stop=0.92)

    def link(self, *args, **kwargs):
        kwargs.update(self.linker_opts)
        return link(args[0], validate_tuple(args[1], 2), *args[2:], **kwargs)

    def link_df(self, *args, **kwargs):
        kwargs.update(self.linker_opts)
        kwargs['diagnostics'] = self.do_diagnostics
        return link_df(*args, **kwargs)

    def link_df_iter(self, *args, **kwargs):
        kwargs.update(self.linker_opts)
        kwargs['diagnostics'] = self.do_diagnostics
        args = list(args)
        features = args.pop(0)
        res = pandas_concat(link_df_iter(
            (df for fr, df in features.groupby('frame')), *args, **kwargs))
        return pandas_sort(res, ['particle', 'frame']).reset_index(drop=True)


class TestOnce(StrictTestCase):
    # simple API tests that need only run on one engine
    def setUp(self):
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        self.features = f

    def test_t_column(self):
        f = self.features.copy()
        cols = list(f.columns)
        name = 'arbitrary name'
        cols[cols.index('frame')] = name
        f.columns = cols

        # smoke tests
        link_df(f, 5, t_column=name, verify_integrity=True)

        f_iter = (frame for fnum, frame in f.groupby('arbitrary name'))
        list(link_df_iter(f_iter, 5, t_column=name, verify_integrity=True))

    def test_check_iter(self):
        """Check that link_df_iter() makes a useful error message if we
        try to pass a single DataFrame."""
        with self.assertRaises(ValueError):
            list(link_df_iter(self.features.copy(), 5))


class SubnetNeededTests(CommonTrackingTests):
    """Tests that assume a best-effort subnet linker (i.e. not "drop")."""
    def test_two_nearby_steppers(self):
        N = 5
        Y = 2
        # Begin second feature one frame later than the first, so the particle labeling (0, 1) is
        # established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1), 'frame': np.arange(1, N)})
        f = pandas_concat([a, b])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([np.zeros(N), np.ones(N - 1)])
        pandas_sort(expected, ['particle', 'frame'], inplace=True)
        actual = self.link_df(f, 5)
        assert_traj_equal(actual, expected)
        actual_iter = self.link_df_iter(f, 5, hash_size=(50, 50))
        assert_traj_equal(actual_iter, expected)

        # Sort rows by frame (normal use)
        actual = self.link_df(pandas_sort(f, 'frame'), 5)
        assert_traj_equal(actual, expected)
        actual_iter = self.link_df_iter(pandas_sort(f, 'frame'), 5, hash_size=(50, 50))
        assert_traj_equal(actual_iter, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link_df(f1, 5)
        assert_traj_equal(actual, expected)
        actual_iter = self.link_df_iter(f1, 5, hash_size=(50, 50))
        assert_traj_equal(actual_iter, expected)

        if self.do_diagnostics:
            assert 'diag_subnet' in self.diag.columns
            assert 'diag_subnet_size' in self.diag.columns
            # Except for frame in which they appear, all particles should have
            # been labeled with a search_range
            assert not any(self.diag['diag_search_range'][
                               actual_iter.frame > 1].isnull())
            # The number of loop iterations is reported by the numba linker only
            if self.linker_opts['link_strategy'] == 'numba':
                assert 'diag_subnet_iterations' in self.diag.columns

    def test_two_nearby_steppers_one_gapped(self):
        N = 5
        Y = 2
        # Begin second feature one frame later than the first, so the particle labeling (0, 1) is
        # established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1), 'frame': np.arange(1, N)})
        a = a.drop(3).reset_index(drop=True)
        f = pandas_concat([a, b])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([np.array([0, 0, 0, 2]), np.ones(N - 1)])
        pandas_sort(expected, ['particle', 'frame'], inplace=True)
        expected.reset_index(drop=True, inplace=True)
        actual = self.link_df(f, 5)
        assert_traj_equal(actual, expected)
        actual_iter = self.link_df_iter(f, 5, hash_size=(50, 50))
        assert_traj_equal(actual_iter, expected)

        # Sort rows by frame (normal use)
        actual = self.link_df(pandas_sort(f, 'frame'), 5)
        assert_traj_equal(actual, expected)
        actual_iter = self.link_df_iter(pandas_sort(f, 'frame'), 5, hash_size=(50, 50))
        assert_traj_equal(actual_iter, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link_df(f1, 5)
        assert_traj_equal(actual, expected)
        actual_iter = self.link_df_iter(f1, 5, hash_size=(50, 50))
        assert_traj_equal(actual_iter, expected)

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
        actual = self.link_df(f, 5)
        assert_traj_equal(actual, expected)
        actual = self.link_df_iter(f, 5, hash_size=(2*M, 2*M + Y))
        assert_traj_equal(actual, expected)

        # Several 2D random walks
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
        actual = self.link_df(f, 5)
        assert_traj_equal(actual, expected)
        actual = self.link_df_iter(f, 5, hash_size=(2*M, 2*M))
        assert_traj_equal(actual, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link_df(f1, 5)
        assert_traj_equal(actual, expected)
        actual = self.link_df_iter(f1, 5, hash_size=(2*M, 2*M))
        assert_traj_equal(actual, expected)

    def test_quadrature_sum(self):
        """A simple test to check whether the subnet linker adds
        distances in quadrature (as in Crocker-Grier)."""
        def subnet_test(epsilon):
            """Returns 2 features in 2 frames, which represent a special
            case when the subnet linker adds distances in quadrature. With
            epsilon=0, subnet linking is degenerate. Therefore
            linking should differ for positive and negative epsilon."""
            return pd.DataFrame([(0, 10, 11), (0, 10, 8),
                                 (1, 9, 10), (1, 12, 10 + epsilon)],
                         columns=['frame', 'x', 'y'])
        trneg = self.link_df(subnet_test(0.01), 5, retain_index=True)
        trpos = self.link_df(subnet_test(-0.01), 5, retain_index=True)
        assert not np.allclose(trneg.particle.values, trpos.particle.values)

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

        actual = self.link_df(f, 13)
        pandas_sort(case1, ['x'], inplace=True)
        pandas_sort(actual, ['x'], inplace=True)
        assert_array_equal(actual['particle'].values.astype(int),
                           case1['particle'].values.astype(int))

        actual = self.link_df(f, 12)
        pandas_sort(case2, ['x'], inplace=True)
        pandas_sort(actual, ['x'], inplace=True)
        assert_array_equal(actual['particle'].values.astype(int),
                           case2['particle'].values.astype(int))

    def test_memory(self):
        """A unit-stepping trajectory and a random walk are observed
        simultaneously. The random walk is missing from one observation."""
        a = [p[0] for p in unit_steps()]
        b = [p[0] for p in random_walk_legacy()]
        # b[2] is intentionally omitted below.
        gapped = lambda: deepcopy([[a[0], b[0]], [a[1], b[1]], [a[2]],
                                   [a[3], b[3]], [a[4], b[4]]])
        safe_disp = 1 + random_x.max() - random_x.min()  # Definitely large enough
        t0 = self.link(gapped(), safe_disp, hash_generator((10, 10), 1), memory=0)
        assert len(t0) == 3, len(t0)
        t2 = self.link(gapped(), safe_disp, hash_generator((10, 10), 1), memory=2)
        assert len(t2) == 2, len(t2)

    def test_memory_removal(self):
        """BUG: A particle remains in memory after its Track is resumed, leaving two
        copies that can independently pick up desinations, leaving two Points in the
        same Track in a single level."""
        levels  = []
        levels.append([PointND(0, [1, 1]), PointND(0, [4, 1])])  # two points
        levels.append([PointND(1, [1, 1])])  # one vanishes, but is remembered
        levels.append([PointND(2, [1, 1]), PointND(2, [2, 1])]) # resume Track
        levels.append([PointND(3, [1, 1]), PointND(3, [2, 1]), PointND(3, [4, 1])])
        t = self.link(levels, 5, hash_generator((10, 10), 1), memory=2)
        assert len(t) == 3, len(t)

    def test_memory_with_late_appearance(self):
        a = [p[0] for p in unit_steps()]
        b = [p[0] for p in random_walk_legacy()]
        gapped = lambda: deepcopy([[a[0]], [a[1], b[1]], [a[2]],
                                   [a[3]], [a[4], b[4]]])
        safe_disp = 1 + random_x.max() - random_x.min()  # large enough
        t0 = self.link(gapped(), safe_disp, hash_generator((10, 10), 1), memory=1)
        assert len(t0) == 3, len(t0)
        t2 = self.link(gapped(), safe_disp, hash_generator((10, 10), 1), memory=4)
        assert len(t2) == 2, len(t2)

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
        actual = self.link_df(f, 5, memory=1)
        assert_traj_equal(actual, expected)
        if self.do_diagnostics:
            assert 'diag_remembered' in self.diag.columns
        actual_iter = self.link_df_iter(f, 5, hash_size=(50, 50), memory=1)
        assert_traj_equal(actual_iter, expected)
        if self.do_diagnostics:
            assert 'diag_remembered' in self.diag.columns

        # Sort rows by frame (normal use)
        actual = self.link_df(pandas_sort(f, 'frame'), 5, memory=1)
        assert_traj_equal(actual, expected)
        if self.do_diagnostics:
            assert 'diag_remembered' in self.diag.columns
        actual_iter = self.link_df_iter(pandas_sort(f, 'frame'), 5,
                                        memory=1, hash_size=(50, 50))
        assert_traj_equal(actual_iter, expected)
        if self.do_diagnostics:
            assert 'diag_remembered' in self.diag.columns

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link_df(f1, 5, memory=1)
        assert_traj_equal(actual, expected)
        if self.do_diagnostics:
            assert 'diag_remembered' in self.diag.columns
        actual_iter = self.link_df_iter(f1, 5, memory=1, hash_size=(50, 50))
        assert_traj_equal(actual_iter, expected)
        if self.do_diagnostics:
            assert 'diag_remembered' in self.diag.columns

    def test_pathological_tracking(self):
        level_count = 5
        p_count = 16
        levels = []
        shift = 1

        for j in range(level_count):
            level = []
            for k in np.arange(p_count) * 2:
                level.append(PointND(k // 2, (j, k + j * shift)))
            levels.append(level)

        hash_generator = lambda: Hash_table((level_count + 1,
                                             p_count*2 + level_count*shift + 1),
                                            .5)
        tracks = self.link(levels, 8, hash_generator)

        assert len(tracks) == p_count, len(tracks)


class DiagnosticsTests(CommonTrackingTests):
    """Mixin to obtain diagnostic info from the linker.

    Makes examining that info optional, so that most tests can focus on
    correctness of tracking.
    """
    do_diagnostics = True

    def _strip_diag(self, df):
        """Move diagnostic columns from the returned DataFrame into a buffer.
        """
        diag_cols = [cn for cn in df.columns if cn.startswith('diag_')]
        self.diag = df.reindex(columns=diag_cols)
        return strip_diagnostics(df)

    def link_df(self, *args, **kwargs):
        return self._strip_diag(
            super().link_df(*args, **kwargs))

    def link_df_iter(self, *args, **kwargs):
        df = self._strip_diag(
            super().link_df_iter(*args, **kwargs))
        # pandas_concat() can mess with the column order if not all columns
        # are present in all DataFrames. So we enforce it here.
        return df.reindex(columns=['frame', 'x', 'y', 'particle'])


class NumbaOnlyTests(SubnetNeededTests):
    """Tests that are unbearably slow without a fast subnet linker."""
    def test_adaptive_range(self):
        cg = contracting_grid()
        # Allow 5 applications of the step
        tracks = self.link_df(cg, 1, adaptive_step=0.8, adaptive_stop=0.32)
        # Transform back to origin
        tracks.x -= 100
        tracks.y -= 100
        assert len(cg) == len(tracks)
        tr0 = tracks[tracks.frame == 0].set_index('particle')
        tr1 = tracks[tracks.frame == 1].set_index('particle')
        only0 = list(set(tr0.index) - set(tr1.index))
        only1 = list(set(tr1.index) - set(tr0.index))
        # From the first frame, the outermost particles should have been lost.
        assert all((tr0.x.loc[only0].abs() > 9.5) | (tr0.y.loc[only0].abs() > 9.5))
        # There should be new tracks in the second frame, corresponding to the
        # middle radii.
        assert all((tr1.x.loc[only1].abs() == 4.5) | (tr1.y.loc[only1].abs() == 4.5))
        if self.do_diagnostics:
            # We use this opportunity to check for diagnostic data
            # made by the numba linker only.
            assert 'diag_subnet_iterations' in self.diag.columns


class TestKDTreeWithDropLink(CommonTrackingTests, StrictTestCase):
    def setUp(self):
        self.linker_opts = dict(link_strategy='drop',
                                neighbor_strategy='KDTree')

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
        without_subnet = self.link_df(f, 1.5, retain_index=True)
        assert_traj_equal(without_subnet, f_expected_without_subnet)
        with_subnet = self.link_df(f, 5, retain_index=True)
        assert set(with_subnet.particle) == {0, 1, 2}


class TestBTreeWithRecursiveLink(SubnetNeededTests, StrictTestCase):
    def setUp(self):
        self.linker_opts = dict(link_strategy='recursive',
                                neighbor_strategy='BTree')


class TestBTreeWithNonrecursiveLink(SubnetNeededTests, StrictTestCase):
    def setUp(self):
        self.linker_opts = dict(link_strategy='nonrecursive',
                                neighbor_strategy='BTree')


class TestBTreeWithNonrecursiveLinkDiag(DiagnosticsTests, TestBTreeWithNonrecursiveLink):
    pass


class TestKDTreeWithRecursiveLink(SubnetNeededTests, StrictTestCase):
    def setUp(self):
        self.linker_opts = dict(link_strategy='recursive',
                                neighbor_strategy='KDTree')


class TestKDTreeWithRecursiveLinkDiag(DiagnosticsTests, TestKDTreeWithRecursiveLink):
    pass


class TestKDTreeWithNonrecursiveLink(SubnetNeededTests, StrictTestCase):
    def setUp(self):
        self.linker_opts = dict(link_strategy='nonrecursive',
                                neighbor_strategy='KDTree')


class TestKDTreeWithNumbaLink(NumbaOnlyTests, StrictTestCase):
    def setUp(self):
        _skip_if_no_numba()
        self.linker_opts = dict(link_strategy='numba',
                                neighbor_strategy='KDTree')


class TestKDTreeWithNumbaLinkDiag(DiagnosticsTests, TestKDTreeWithNumbaLink):
    pass


class TestBTreeWithNumbaLink(NumbaOnlyTests, StrictTestCase):
    def setUp(self):
        _skip_if_no_numba()
        self.linker_opts = dict(link_strategy='numba',
                                neighbor_strategy='BTree')


if __name__ == '__main__':
    import unittest
    unittest.main()
