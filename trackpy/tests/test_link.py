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
                                 assert_almost_equal, assert_index_equal)

path, _ = os.path.split(os.path.abspath(__file__))
path = os.path.join(path, 'data')

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


def _skip_if_no_pytables():
    try:
        import tables
    except ImportError:
        raise nose.SkipTest('PyTables not installed. Skipping.')


def random_walk(N):
    return np.cumsum(np.random.randn(N))

class CommonTrackingTests(object):

    def test_one_trivial_stepper(self):
        # One 1D stepper
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        actual = self.link(f, 5)
        expected = f.copy()
        expected['probe'] = np.zeros(N)
        assert_frame_equal(actual, expected)

    def test_two_isolated_steppers(self):
        N = 5
        Y = 25
        # Begin second feature one frame later than the first, so the probe labeling (0, 1) is
        # established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1), 'frame': np.arange(1, N)})
        f = pd.concat([a, b])
        actual = self.link(f, 5)
        expected = f.copy().reset_index(drop=True)
        expected['probe'] = np.concatenate([np.zeros(N), np.ones(N - 1)])
        expected.sort(['probe', 'frame'], inplace=True)
        assert_frame_equal(actual, expected)

        # Sort rows by frame (normal use)
        actual = self.link(f.sort('frame'), 5)
        assert_frame_equal(actual, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link(f1, 5)
        assert_frame_equal(actual, expected)

    def test_two_isolated_steppers_one_gapped(self):
        N = 5
        Y = 25
        # Begin second feature one frame later than the first, 
        # so the probe labeling (0, 1) is established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N), 
                      'frame': np.arange(N)})
        a = a.drop(3).reset_index(drop=True)
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1), 
                      'frame': np.arange(1, N)})
        f = pd.concat([a, b])
        expected = f.copy()
        actual = self.link(f, 5)
        expected['probe'] = np.concatenate([np.array([0, 0, 0, 2]), np.ones(N - 1)])
        expected.sort(['probe', 'frame'], inplace=True)
        expected.reset_index(drop=True, inplace=True)
        print expected
        assert_frame_equal(actual, expected)

        # Sort rows by frame (normal use)
        actual = self.link(f.sort('frame'), 5)
        assert_frame_equal(actual, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link(f1, 5)
        assert_frame_equal(actual, expected)

    def test_two_nearby_steppers(self):
        N = 5
        Y = 2
        # Begin second feature one frame later than the first, so the probe labeling (0, 1) is
        # established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1), 'frame': np.arange(1, N)})
        f = pd.concat([a, b])
        expected = f.copy().reset_index(drop=True)
        expected['probe'] = np.concatenate([np.zeros(N), np.ones(N - 1)])
        expected.sort(['probe', 'frame'], inplace=True)
        actual = self.link(f, 5)
        assert_frame_equal(actual, expected)

        # Sort rows by frame (normal use)
        actual = self.link(f.sort('frame'), 5)
        assert_frame_equal(actual, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link(f1, 5)
        assert_frame_equal(actual, expected)

    def test_two_nearby_steppers_one_gapped(self):
        N = 5
        Y = 2
        # Begin second feature one frame later than the first, so the probe labeling (0, 1) is
        # established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1), 'frame': np.arange(1, N)})
        a = a.drop(3).reset_index(drop=True)
        f = pd.concat([a, b])
        expected = f.copy().reset_index(drop=True)
        expected['probe'] = np.concatenate([np.array([0, 0, 0, 2]), np.ones(N - 1)])
        expected.sort(['probe', 'frame'], inplace=True)
        expected.reset_index(drop=True, inplace=True)
        actual = self.link(f, 5)
        assert_frame_equal(actual, expected)

        # Sort rows by frame (normal use)
        actual = self.link(f.sort('frame'), 5)
        assert_frame_equal(actual, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link(f1, 5)
        assert_frame_equal(actual, expected)
 
    def test_memory_on_one_gap(self):
        N = 5
        Y = 2
        # Begin second feature one frame later than the first, so the probe labeling (0, 1) is
        # established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1), 'frame': np.arange(1, N)})
        a = a.drop(3).reset_index(drop=True)
        f = pd.concat([a, b])
        actual = self.link(f, 5, memory=1)
        expected = f.copy().reset_index(drop=True)
        expected['probe'] = np.concatenate([np.array([0, 0, 0, 0]), np.ones(N - 1)])
        expected.sort(['probe', 'frame'], inplace=True)
        expected.reset_index(drop=True, inplace=True)
        assert_frame_equal(actual, expected)

        # Sort rows by frame (normal use)
        actual = self.link(f.sort('frame'), 5, memory=1)
        assert_frame_equal(actual, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link(f1, 5, memory=1)
        assert_frame_equal(actual, expected)

    def test_isolated_continuous_random_walks(self):
        # Two 2D random walks
        np.random.seed(0)
        N = 30
        Y = 250
        M = 20 # margin, because negative values raise OutOfHash
        a = DataFrame({'x': M + random_walk(N), 'y': M + random_walk(N), 'frame': np.arange(N)})
        b = DataFrame({'x': M + random_walk(N - 1), 'y': M + Y + random_walk(N - 1), 'frame': np.arange(1, N)})
        f = pd.concat([a, b])
        actual = self.link(f, 5)
        expected = f.copy().reset_index(drop=True)
        expected['probe'] = np.concatenate([np.zeros(N), np.ones(N - 1)])
        expected.sort(['probe', 'frame'], inplace=True)
        assert_frame_equal(actual, expected)

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
        actual = self.link(f, 5)
        expected = f.copy().reset_index(drop=True)
        expected['probe'] = np.concatenate([i*np.ones(N - i) for i in range(len(initial_positions))])
        expected.sort(['probe', 'frame'], inplace=True)
        assert_frame_equal(actual, expected)

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
        actual = self.link(f, 5)
        expected = f.copy().reset_index(drop=True)
        expected['probe'] = np.concatenate([np.zeros(N), np.ones(N - 1)])
        expected.sort(['probe', 'frame'], inplace=True)
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
        actual = self.link(f, 5)
        expected = f.copy().reset_index(drop=True)
        expected['probe'] = np.concatenate([i*np.ones(N - i) for i in range(len(initial_positions))])
        expected.sort(['probe', 'frame'], inplace=True)
        assert_frame_equal(actual, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link(f1, 5)
        assert_frame_equal(actual, expected)

    def test_start_at_frame_other_than_zero(self):
        # One 1D stepper
        N = 5
        FIRST_FRAME = 3
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N), 
                      'frame': FIRST_FRAME + np.arange(N)})
        actual = self.link(f, 5)
        expected = f.copy()
        expected['probe'] = np.zeros(N)
        assert_frame_equal(actual, expected)

    def test_blank_frame_no_memory(self):
        # One 1D stepper
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N),
                      'frame': [0, 1, 2, 4, 5]})
        actual = self.link(f, 5)
        expected = f.copy()
        expected['probe'] = np.zeros(N)
        assert_frame_equal(actual, expected)
        # This doesn't error, but we might wish it would
        # give the probe a new ID after the gap. It just
        # ignores the missing frame.

    def test_real_data_that_causes_duplicate_bug(self):
        filename = 'reproduce_duplicate_track_assignment.df'
        f = pd.read_pickle(os.path.join(path, filename))
        self.link(f, 8, 2) # not all parameters reproduce it, but these do
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
    
        for t in tracks:
            x, y = zip(*[p.pos for p in t])
            dx = np.diff(x)
            dy = np.diff(y)
    
    # THIS DOES NOT WORK BECAUSE ORDER IS ARBIRARY!
    # Think of way to make this into a working test someday....
    # def test_real_data(self):
    #     load = lambda filename: pd.read_pickle(os.path.join(path, filename))
    #     features = load('features_size9_masscut2000.df')
    #     traj_no_memory = load('traj_search5_memory0.df')
    #     traj_memory = load('traj_search5_memory2.df')

    #     actual = self.link(features, 5, memory=0)
    #     assert_allclose(actual, traj_no_memory, atol=0.1)
    #     actual = self.link(features, 5, memory=2)
    #     assert_allclose(actual, traj_memory, atol=0.1)

#class TestKDTreeTracking(CommonTrackingTests, unittest.TestCase):
#    def setUp(self):
#        self.link = tp.linking_experimental.link

class TestHashTableWithRecursiveLink(CommonTrackingTests, unittest.TestCase):
    # This is the default.
    def setUp(self):
        self.link = tp.link_df

class TestHasTableWithNonrecursiveLink(CommonTrackingTests, unittest.TestCase):
    def setUp(self):
        def curried_link(*args, **kwargs):
            kwargs['link_strategy'] = 'nonrecursive'
            return tp.link_df(*args, **kwargs)
        self.link = curried_link


# Removed after trackpy refactor -- restore with new API.
# class TestLinkOnDisk(unittest.TestCase):
# 
#    def setUp(self):
#        _skip_if_no_pytables()
#        filename = os.path.join(path, 'features_size9_masscut2000.df')
#        f = pd.read_pickle(filename)
#        self.key = 'features'
#        with pd.get_store('temp1.h5') as store:
#            store.put(self.key, f)
#        with pd.get_store('temp2.h5') as store:
#            store.append(self.key, f, data_columns=['frame'])
#
#    def test_nontabular_raises(self):
#        # Attempting to Link a non-tabular node should raise.
#        _skip_if_no_pytables()
#        f = lambda: tp.LinkOnDisk('temp1.h5', self.key)
#        self.assertRaises(ValueError, f)
#
#    def test_nontabular_with_use_tabular_copy(self):
#        # simple smoke test
#        _skip_if_no_pytables()
#        linker = tp.LinkOnDisk('temp1.h5', self.key, use_tabular_copy=True)
#        linker.link(8, 2)
#        linker.save('temp3.h5', 'traj')
#
#    def test_tabular(self):
#        # simple smoke test
#        _skip_if_no_pytables()
#        linker = tp.LinkOnDisk('temp2.h5', self.key)
#        linker.link(8, 2)
#        linker.save('temp4.h5', 'traj')
#
#    def tearDown(self):
#        temp_files = ['temp1.h5', 'temp2.h5', 'temp3.h5', 'temp4.h5']
#        for filename in temp_files:
#            try:
#                os.remove(filename)
#            except OSError:
#                pass
