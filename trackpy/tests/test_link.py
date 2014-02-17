from __future__ import division
import os
import trackpy as tp
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import unittest
import nose
from numpy.testing import assert_almost_equal, assert_allclose
from numpy.testing.decorators import slow
from pandas.util.testing import (assert_series_equal, assert_frame_equal,
                                 assert_almost_equal)

path, _ = os.path.split(os.path.abspath(__file__))
path = os.path.join(path, 'data')

from trackpy.try_numba import NUMBA_AVAILABLE
from trackpy.linking import PointND, link, Hash_table
from copy import deepcopy

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
        raise nose.SkipTest('numba not installed. Skipping.')


def random_walk(N):
    return np.cumsum(np.random.randn(N))

class CommonTrackingTests(object):

    def test_one_trivial_stepper(self):
        # One 1D stepper
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        expected = f.copy()
        expected['particle'] = np.zeros(N)
        actual = self.link_df(f, 5)
        assert_frame_equal(actual, expected)
        actual_iter = self.link_df_iter(f, 5, hash_size=(10, 2))
        assert_frame_equal(actual_iter, expected)

    def test_two_isolated_steppers(self):
        N = 5
        Y = 25
        # Begin second feature one frame later than the first, so the particle labeling (0, 1) is
        # established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1), 'frame': np.arange(1, N)})
        f = pd.concat([a, b])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([np.zeros(N), np.ones(N - 1)])
        expected.sort(['particle', 'frame'], inplace=True)
        actual = self.link_df(f, 5)
        assert_frame_equal(actual, expected)
        actual_iter = self.link_df_iter(f, 5, hash_size=(50, 50))
        assert_frame_equal(actual_iter, expected)

        # Sort rows by frame (normal use)
        actual = self.link_df(f.sort('frame'), 5)
        assert_frame_equal(actual, expected)
        actual_iter = self.link_df_iter(f.sort('frame'), 5, hash_size=(50, 50))
        assert_frame_equal(actual_iter, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link_df(f1, 5)
        assert_frame_equal(actual, expected)
        actual_iter = self.link_df_iter(f1, 5, hash_size=(50, 50))
        assert_frame_equal(actual_iter, expected)

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
        f = pd.concat([a, b])
        expected = f.copy()
        expected['particle'] = np.concatenate([np.array([0, 0, 0, 2]), np.ones(N - 1)])
        expected.sort(['particle', 'frame'], inplace=True)
        expected.reset_index(drop=True, inplace=True)
        actual = self.link_df(f, 5)
        assert_frame_equal(actual, expected)
        actual_iter = self.link_df_iter(f, 5, hash_size=(50, 50))
        assert_frame_equal(actual_iter, expected)
        # link_df_iter() tests not performed, because hash_size is
        # not knowable from the first frame alone.

        # Sort rows by frame (normal use)
        actual = self.link_df(f.sort('frame'), 5)
        assert_frame_equal(actual, expected)
        actual_iter = self.link_df_iter(f.sort('frame'), 5, hash_size=(50, 50))
        assert_frame_equal(actual_iter, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link_df(f1, 5)
        assert_frame_equal(actual, expected)
        actual_iter = self.link_df_iter(f1, 5, hash_size=(50, 50))
        assert_frame_equal(actual_iter, expected)

    def test_two_nearby_steppers(self):
        N = 5
        Y = 2
        # Begin second feature one frame later than the first, so the particle labeling (0, 1) is
        # established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1), 'frame': np.arange(1, N)})
        f = pd.concat([a, b])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([np.zeros(N), np.ones(N - 1)])
        expected.sort(['particle', 'frame'], inplace=True)
        actual = self.link_df(f, 5)
        assert_frame_equal(actual, expected)
        actual_iter = self.link_df_iter(f, 5, hash_size=(50, 50))
        assert_frame_equal(actual_iter, expected)

        # Sort rows by frame (normal use)
        actual = self.link_df(f.sort('frame'), 5)
        assert_frame_equal(actual, expected)
        actual_iter = self.link_df_iter(f.sort('frame'), 5, hash_size=(50, 50))
        assert_frame_equal(actual_iter, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link_df(f1, 5)
        assert_frame_equal(actual, expected)
        actual_iter = self.link_df_iter(f1, 5, hash_size=(50, 50))
        assert_frame_equal(actual_iter, expected)

    def test_two_nearby_steppers_one_gapped(self):
        N = 5
        Y = 2
        # Begin second feature one frame later than the first, so the particle labeling (0, 1) is
        # established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1), 'frame': np.arange(1, N)})
        a = a.drop(3).reset_index(drop=True)
        f = pd.concat([a, b])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([np.array([0, 0, 0, 2]), np.ones(N - 1)])
        expected.sort(['particle', 'frame'], inplace=True)
        expected.reset_index(drop=True, inplace=True)
        actual = self.link_df(f, 5)
        assert_frame_equal(actual, expected)
        actual_iter = self.link_df_iter(f, 5, hash_size=(50, 50))
        assert_frame_equal(actual_iter, expected)

        # Sort rows by frame (normal use)
        actual = self.link_df(f.sort('frame'), 5)
        assert_frame_equal(actual, expected)
        actual_iter = self.link_df_iter(f.sort('frame'), 5, hash_size=(50, 50))
        assert_frame_equal(actual_iter, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link_df(f1, 5)
        assert_frame_equal(actual, expected)
        actual_iter = self.link_df_iter(f1, 5, hash_size=(50, 50))
        assert_frame_equal(actual_iter, expected)

    def test_memory_on_one_gap(self):
        N = 5
        Y = 2
        # Begin second feature one frame later than the first, so the particle labeling (0, 1) is
        # established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1), 'frame': np.arange(1, N)})
        a = a.drop(3).reset_index(drop=True)
        f = pd.concat([a, b])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([np.array([0, 0, 0, 0]), np.ones(N - 1)])
        expected.sort(['particle', 'frame'], inplace=True)
        expected.reset_index(drop=True, inplace=True)
        actual = self.link_df(f, 5, memory=1)
        assert_frame_equal(actual, expected)
        actual_iter = self.link_df_iter(f, 5, hash_size=(50, 50), memory=1)
        assert_frame_equal(actual_iter, expected)

        # Sort rows by frame (normal use)
        actual = self.link_df(f.sort('frame'), 5, memory=1)
        assert_frame_equal(actual, expected)
        actual_iter = self.link_df_iter(f.sort('frame'), 5,
                                        memory=1, hash_size=(50, 50))
        assert_frame_equal(actual_iter, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link_df(f1, 5, memory=1)
        assert_frame_equal(actual, expected)
        actual_iter = self.link_df_iter(f1, 5, memory=1, hash_size=(50, 50))
        assert_frame_equal(actual_iter, expected)

    def test_isolated_continuous_random_walks(self):
        # Two 2D random walks
        np.random.seed(0)
        N = 30
        Y = 250
        M = 20 # margin, because negative values raise OutOfHash
        a = DataFrame({'x': M + random_walk(N), 'y': M + random_walk(N), 'frame': np.arange(N)})
        b = DataFrame({'x': M + random_walk(N - 1), 'y': M + Y + random_walk(N - 1), 'frame': np.arange(1, N)})
        f = pd.concat([a, b])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([np.zeros(N), np.ones(N - 1)])
        expected.sort(['particle', 'frame'], inplace=True)
        actual = self.link_df(f, 5)
        assert_frame_equal(actual, expected)
        actual_iter = self.link_df_iter(f, 5, hash_size=(2*M, Y + 2*M))
        assert_frame_equal(actual_iter, expected)

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
        f = pd.concat([walk(*pos) for pos in initial_positions])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([i*np.ones(N - i) for i in range(len(initial_positions))])
        expected.sort(['particle', 'frame'], inplace=True)
        actual = self.link_df(f, 5)
        assert_frame_equal(actual, expected)
        actual_iter = self.link_df_iter(f, 5, hash_size=(200 + M, 200 + M))
        assert_frame_equal(actual_iter, expected)

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
        f = pd.concat([a, b])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([np.zeros(N), np.ones(N - 1)])
        expected.sort(['particle', 'frame'], inplace=True)
        actual = self.link_df(f, 5)
        assert_frame_equal(actual, expected)
        actual = self.link_df_iter(f, 5, hash_size=(2*M, 2*M + Y))
        assert_frame_equal(actual, expected)

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
        f = pd.concat([walk(*pos) for pos in initial_positions])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([i*np.ones(N - i) for i in range(len(initial_positions))])
        expected.sort(['particle', 'frame'], inplace=True)
        actual = self.link_df(f, 5)
        assert_frame_equal(actual, expected)
        actual = self.link_df_iter(f, 5, hash_size=(2*M, 2*M))
        assert_frame_equal(actual, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link_df(f1, 5)
        assert_frame_equal(actual, expected)
        actual = self.link_df_iter(f1, 5, hash_size=(2*M, 2*M))
        assert_frame_equal(actual, expected)

    def test_start_at_frame_other_than_zero(self):
        # One 1D stepper
        N = 5
        FIRST_FRAME = 3
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N), 
                      'frame': FIRST_FRAME + np.arange(N)})
        expected = f.copy()
        expected['particle'] = np.zeros(N)
        actual = self.link_df(f, 5)
        assert_frame_equal(actual, expected)
        actual = self.link_df_iter(f, 5, hash_size=(6, 2))
        assert_frame_equal(actual, expected)

    def test_blank_frame_no_memory(self):
        # One 1D stepper
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N),
                      'frame': [0, 1, 2, 4, 5]})
        expected = f.copy()
        expected['particle'] = np.zeros(N)
        actual = self.link_df(f, 5)
        assert_frame_equal(actual, expected)
        actual = self.link_df_iter(f, 5, hash_size=(10, 10))
        assert_frame_equal(actual, expected)
        # This doesn't error, but we might wish it would
        # give the particle a new ID after the gap. It just
        # ignores the missing frame.

    def test_real_data_that_causes_duplicate_bug(self):
        filename = 'reproduce_duplicate_track_assignment.df'
        f = pd.read_pickle(os.path.join(path, filename))
        self.link_df(f, 8, 2) # not all parameters reproduce it, but these do
        # If there are duplicates, _verify_integrity will raise and this will error.

    def test_search_range(self):
        t = link(unit_steps(), 1.1, hash_generator((10, 10), 1))
        assert len(t) == 1  # One track
        t_short = link(unit_steps(), 0.9, hash_generator((10, 10), 1))
        assert len(t_short) == len(unit_steps())  # Each step is a separate track.

        t = link(random_walk_legacy(), max_disp + 0.1, 
                 hash_generator((10, 10), 1))
        assert len(t) == 1  # One track
        t_short = link(random_walk_legacy(), max_disp - 0.1, 
                       hash_generator((10, 10), 1))
        assert len(t_short) > 1  # Multiple tracks


    def test_memory(self):
        """A unit-stepping trajectory and a random walk are observed
        simultaneously. The random walk is missing from one observation."""
        a = [p[0] for p in unit_steps()]
        b = [p[0] for p in random_walk_legacy()]
        # b[2] is intentionally omitted below.
        gapped = lambda: deepcopy([[a[0], b[0]], [a[1], b[1]], [a[2]],
                                  [a[3], b[3]], [a[4], b[4]]])
        safe_disp = 1 + random_x.max() - random_x.min()  # Definitely large enough
        t0 = link(gapped(), safe_disp, hash_generator((10, 10), 1), memory=0)
        assert len(t0) == 3, len(t0)
        t2 = link(gapped(), safe_disp, hash_generator((10, 10), 1), memory=2)
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
        t = link(levels, 5, hash_generator((10, 10), 1), memory=2)
        assert len(t) == 3, len(t)
    
     
    def test_memory_with_late_appearance(self):
        a = [p[0] for p in unit_steps()]
        b = [p[0] for p in random_walk_legacy()]
        gapped = lambda: deepcopy([[a[0]], [a[1], b[1]], [a[2]],
                                  [a[3]], [a[4], b[4]]])
        safe_disp = 1 + random_x.max() - random_x.min()  # large enough
        t0 = link(gapped(), safe_disp, hash_generator((10, 10), 1), memory=1)
        assert len(t0) == 3, len(t0)
        t2 = link(gapped(), safe_disp, hash_generator((10, 10), 1), memory=4)
        assert len(t2) == 2, len(t2)
    
    
    def test_box_size(self):
        """No matter what the box size, there should be one track, and it should
        contain all the points."""
        for box_size in [0.1, 1, 10]:
            t1 = link(unit_steps(), 1.1, hash_generator((10, 10), box_size))
            t2 = link(random_walk_legacy(), max_disp + 1,
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
        tracks = link(levels, 1.5, hash_generator)
    
        assert len(tracks) == p_count
    
        for t in tracks:
            x, y = zip(*[p.pos for p in t])
            dx = np.diff(x)
            dy = np.diff(y)
    
            assert np.sum(dx) == level_count - 1
            assert np.sum(dy) == 0
    
    
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
        tracks = link(levels, 8, hash_generator)
    
        assert len(tracks) == p_count, len(tracks)

    def link(self, *args, **kwargs):
        kwargs.update(self.linker_opts)
        return tp.link(*args, **kwargs)

    def link_df(self, *args, **kwargs):
        kwargs.update(self.linker_opts)
        return tp.link_df(*args, **kwargs)

    def link_df_iter(self, *args, **kwargs):
        kwargs.update(self.linker_opts)
        args = list(args)
        features = args.pop(0)
        res = pd.concat(tp.link_df_iter(
            (df for fr, df in features.groupby('frame')), *args, **kwargs))
        return res.sort(['particle', 'frame']).reset_index(drop=True)


class TestBTreeWithRecursiveLink(CommonTrackingTests, unittest.TestCase):
    def setUp(self):
        self.linker_opts = dict(link_strategy='recursive',
                                neighbor_strategy='BTree')

class TestBTreeWithNonrecursiveLink(CommonTrackingTests, unittest.TestCase):
    def setUp(self):
        self.linker_opts = dict(link_strategy='nonrecursive',
                                neighbor_strategy='BTree')

class TestKDTreeWithRecursiveLink(CommonTrackingTests, unittest.TestCase):
    def setUp(self):
        self.linker_opts = dict(link_strategy='recursive',
                                neighbor_strategy='KDTree')

class TestKDTreeWithNonrecursiveLink(CommonTrackingTests, unittest.TestCase):
    def setUp(self):
        self.linker_opts = dict(link_strategy='nonrecursive',
                                neighbor_strategy='KDTree')

class TestKDTreeWithNumbaLink(CommonTrackingTests, unittest.TestCase):
    def setUp(self):
        _skip_if_no_numba()
        self.linker_opts = dict(link_strategy='numba',
                                neighbor_strategy='KDTree')

class TestBTreeWithNumbaLink(CommonTrackingTests, unittest.TestCase):
    def setUp(self):
        _skip_if_no_numba()
        self.linker_opts = dict(link_strategy='numba',
                                neighbor_strategy='BTree')

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
