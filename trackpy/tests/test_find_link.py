## TESTS PARTIALLY COPIED FROM trackpy.tests.test_link.py

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import itertools
import numpy as np
import pandas as pd
from pandas import DataFrame
import nose
from numpy.testing import assert_equal

from trackpy.try_numba import NUMBA_AVAILABLE
from trackpy.linking import SubnetOversizeException
from trackpy.utils import pandas_sort
from trackpy.artificial import CoordinateReader
from trackpy.find_link import find_link, link_simple, link_simple_iter, link_simple_df_iter
from trackpy.tests.common import assert_traj_equal, StrictTestCase
from trackpy.tests.test_link import random_walk


def _skip_if_no_numba():
    if not NUMBA_AVAILABLE:
        raise nose.SkipTest('numba not installed. Skipping.')

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
    allpts = pd.concat([pts0, pts1], ignore_index=True)
    allpts.x += 200  # Because BTree doesn't allow negative coordinates
    allpts.y += 200
    return allpts


class SimpleLinkingTests(StrictTestCase):
    def setUp(self):
        self.linker_opts = dict()

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
        f['frame'] = f['frame'].astype(np.int)
        actual = self.link(f, 5)

        # Particle and frame columns should be integer typed
        assert np.issubdtype(actual['particle'], np.int)
        assert np.issubdtype(actual['frame'], np.int)

        # Float-typed input
        f['frame'] = f['frame'].astype(np.float)
        actual = self.link(f, 5)

        # Particle and frame columns should be integer typed
        assert np.issubdtype(actual['particle'], np.int)
        assert np.issubdtype(actual['frame'], np.int)

    def test_two_isolated_steppers(self):
        N = 5
        Y = 25
        # Begin second feature one frame later than the first, so the particle
        # labeling (0, 1) is established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1), 'frame': np.arange(1, N)})
        f = pd.concat([a, b])
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
        f = pd.concat([a, b])
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
        f = pd.concat([a, b])
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
        f = pd.concat([walk(*pos) for pos in initial_positions])
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

    @nose.tools.raises(SubnetOversizeException)
    def test_oversize_fail(self):
        df = contracting_grid()
        self.link(df, search_range=2)

    @nose.tools.raises(SubnetOversizeException)
    def test_adaptive_fail(self):
        """Check recursion limit"""
        self.link(contracting_grid(), search_range=2, adaptive_stop=1.84)

    def test_two_nearby_steppers(self):
        N = 5
        Y = 2
        # Begin second feature one frame later than the first, so the particle
        # labeling (0, 1) is established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1), 'frame': np.arange(1, N)})
        f = pd.concat([a, b])
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
        f = pd.concat([a, b])
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
        f = pd.concat([a, b])
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
        f = pd.concat([walk(*pos) for pos in initial_positions])
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
            (tr0.x.ix[only0].abs() > 19) | (tr0.y.ix[only0].abs() > 19))
        # There should be new tracks in the second frame, corresponding to the
        # middle radii.
        assert all(
            (tr1.x.ix[only1].abs() == 9) | (tr1.y.ix[only1].abs() == 9))

    def link(self, f, search_range, *args, **kwargs):
        kwargs = dict(self.linker_opts, **kwargs)
        return link_simple(f, search_range, *args, **kwargs)


class SimpleLinkingTestsIter(SimpleLinkingTests):
    def link(self, f, search_range, *args, **kwargs):

        def f_iter(f, first_frame, last_frame):
            """ link_iter requires an (optionally enumerated) generator of
            ndarrays """
            for t in np.arange(first_frame, last_frame + 1,
                               dtype=f['frame'].dtype):
                f_filt = f[f['frame'] == t]
                yield t, f_filt[['y', 'x']].values

        res = f.copy()
        res['particle'] = -1
        for t, ids in link_simple_iter(f_iter(f, 0, int(f['frame'].max())),
                                       search_range, *args, **kwargs):
            res.loc[res['frame'] == t, 'particle'] = ids
        return pandas_sort(res, ['particle', 'frame']).reset_index(drop=True)

    def test_output_dtypes(self):
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N),
                       'frame': np.arange(N)})
        # Integer-typed input
        f['frame'] = f['frame'].astype(np.int)
        actual = self.link(f, 5)

        # Particle and frame columns should be integer typed
        assert np.issubdtype(actual['particle'], np.int)
        assert np.issubdtype(actual['frame'], np.int)

        # Float-typed input: frame column type is propagated in link_iter
        f['frame'] = f['frame'].astype(np.float)
        actual = self.link(f, 5)
        assert np.issubdtype(actual['particle'], np.int)
        assert np.issubdtype(actual['frame'], np.float)


class SimpleLinkingTestsDfIter(SimpleLinkingTests):
    def link(self, f, search_range, *args, **kwargs):

        def df_iter(f, first_frame, last_frame):
            """ link_df_iter requires a generator of dataframes """
            for t in range(first_frame, last_frame + 1):
                yield f[f['frame'] == t]

        res_iter = link_simple_df_iter(df_iter(f, 0, int(f['frame'].max())),
                                       search_range, *args, **kwargs)
        res = pd.concat(res_iter)
        return pandas_sort(res, ['particle', 'frame']).reset_index(drop=True)

    def test_output_dtypes(self):
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N),
                       'frame': np.arange(N)})
        # Integer-typed input
        f['frame'] = f['frame'].astype(np.int)
        actual = self.link(f, 5)

        # Particle and frame columns should be integer typed
        assert np.issubdtype(actual['particle'], np.int)
        assert np.issubdtype(actual['frame'], np.int)

        # Float-typed input: frame column type is propagated in link_df_iter
        f['frame'] = f['frame'].astype(np.float)
        actual = self.link(f, 5)
        assert np.issubdtype(actual['particle'], np.int)
        assert np.issubdtype(actual['frame'], np.float)


class FindLinkTests(SimpleLinkingTests):
    def setUp(self):
        super(FindLinkTests, self).setUp()
        self.linker_opts['separation'] = 10
        self.linker_opts['diameter'] = 15

    def link(self, f, search_range, *args, **kwargs):
        # the minimal spacing between features in f is assumed to be 1.

        # from scipy.spatial import cKDTree
        # mindist = 1e7
        # for _, _f in f.groupby('frame'):
        #     dists, _ = cKDTree(_f[['y', 'x']].values).query(_f[['y', 'x']].values, k=2)
        #     mindist = min(mindist, dists[:, 1].min())
        # print("Minimal dist is {0:.3f}".format(mindist))

        kwargs = dict(self.linker_opts, **kwargs)
        size = 3
        separation = kwargs['separation']
        f = f.copy()
        f[['y', 'x']] *= separation
        topleft = (f[['y', 'x']].min().values - 4 * separation).astype(np.int)
        f[['y', 'x']] -= topleft
        shape = (f[['y', 'x']].max().values + 4 * separation).astype(np.int)
        reader = CoordinateReader(f, shape, size)
        if kwargs.get('adaptive_stop', None) is not None:
            kwargs['adaptive_stop'] *= separation
        result = find_link(reader,
                           search_range=search_range*separation,
                           *args, **kwargs)
        result = pandas_sort(result, ['particle', 'frame']).reset_index(drop=True)
        result[['y', 'x']] += topleft
        result[['y', 'x']] /= separation
        return result

    def test_args_dtype(self):
        """Check whether find_link accepts float typed arguments"""
        # One 1D stepper
        N = 5
        f = DataFrame(
            {'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        expected = f.copy()
        expected['particle'] = np.zeros(N)

        # Should not raise
        actual = self.link(f, 5.2, separation=9.5, diameter=15.2)
        assert_traj_equal(actual, expected)


class FindLinkOneFailedFindTests(FindLinkTests):
    def setUp(self):
        super(FindLinkOneFailedFindTests, self).setUp()
        FAIL_FRAME = dict(test_isolated_continuous_random_walks=5,
                          test_nearby_continuous_random_walks=10,
                          test_start_at_frame_other_than_zero=4,
                          test_two_nearby_steppers_one_gapped=2,
                          test_two_isolated_steppers_one_gapped=2)

        test_name = self.id()[self.id().rfind('.') + 1:]
        fail_frame = FAIL_FRAME.get(test_name, 3)

        def callback(image, coords, **unused_kwargs):
            if image.frame_no == fail_frame:
                return np.empty((0, 2))
            else:
                return coords
        self.linker_opts['before_link'] = callback


class FindLinkManyFailedFindTests(FindLinkTests):
    def setUp(self):
        super(FindLinkManyFailedFindTests, self).setUp()
        FAIL_FRAME = dict(test_isolated_continuous_random_walks=5,
                          test_nearby_continuous_random_walks=10,
                          test_start_at_frame_other_than_zero=4,
                          test_two_nearby_steppers_one_gapped=5,
                          test_two_isolated_steppers_one_gapped=5,
                          test_blank_frame_no_memory=5)

        test_name = self.id()[self.id().rfind('.') + 1:]
        fail_frame = FAIL_FRAME.get(test_name, 3)

        def callback(image, coords, **unused_kwargs):
            if image.frame_no >= fail_frame:
                return np.empty((0, 2))
            else:
                return coords
        self.linker_opts['before_link'] = callback


class FindLinkSpecialCases(StrictTestCase):
    # also, for paper images
    do_diagnostics = False  # Don't ask for diagnostic info from linker
    def setUp(self):
        self.linker_opts = dict()
        self.search_range = 12
        self.separation = 4
        self.diameter = 12  # only for characterization
        self.size = 3

    def link(self, f, shape, remove=None, **kwargs):
        _kwargs = dict(diameter=self.diameter,
                       search_range=self.search_range,
                       separation=self.separation)
        _kwargs.update(kwargs)
        if remove is not None:
            callback_coords = f.loc[f['frame'] == 1, ['y', 'x']].values
            remove = np.array(remove, dtype=np.int)
            if np.any(remove < 0) or np.any(remove > len(callback_coords)):
                raise RuntimeError('Invalid test: `remove` is out of bounds.')
            callback_coords = np.delete(callback_coords, remove, axis=0)
            def callback(image, coords, **unused_kwargs):
                if image.frame_no == 1:
                    return callback_coords
                else:
                    return coords
        else:
            callback = None
        reader = CoordinateReader(f, shape, self.size)
        return find_link(reader, before_link=callback, **_kwargs)

    def test_in_margin(self):
        expected = DataFrame({'x': [12, 6], 'y': [12, 5],
                              'frame': [0, 1], 'particle': [0, 0]})

        actual = self.link(expected, shape=(24, 24))
        assert_equal(len(actual), 1)

    def test_one(self):
        expected = DataFrame({'x': [8, 16], 'y': [16, 16],
                              'frame': [0, 1], 'particle': [0, 0]})

        actual = self.link(expected, shape=(24, 24), remove=[0])
        assert_traj_equal(actual, expected)
        actual = self.link(expected, shape=(24, 24), search_range=7, remove=[0])
        assert_equal(len(actual), 1)

    def test_two_isolated(self):
        shape = (32, 32)
        expected = DataFrame({'x': [8, 16, 24, 16], 'y': [8, 8, 24, 24],
                                 'frame': [0, 1, 0, 1], 'particle': [0, 0, 1, 1]})
        for remove in [[], [0], [1], [0, 1]]:
            actual = self.link(expected, shape=shape, remove=remove)
            assert_traj_equal(actual, expected)

    def test_two_single_overlap(self):
        shape = (16, 40)
        # a --> b  c --> d    : b-c overlap
        expected = DataFrame({'x': [8, 16, 24, 32], 'y': [8, 8, 8, 8],
                              'frame': [0, 1, 0, 1], 'particle': [0, 0, 1, 1]})
        for remove in [[], [0], [1], [0, 1]]:
            actual = self.link(expected, shape=shape, remove=remove)
            assert_traj_equal(actual, expected)
        # a --> b  d <-- c    : b-d overlap
        expected = DataFrame({'x': [8, 16, 24, 32], 'y': [8, 8, 8, 8],
                              'frame': [0, 1, 1, 0], 'particle': [0, 0, 1, 1]})
        for remove in [[], [0], [1], [0, 1]]:
            actual = self.link(expected, shape=shape, remove=remove)
            assert_traj_equal(actual, expected)
        # b <-- a  d --> c    : a-d overlap
        expected = DataFrame({'x': [8, 16, 24, 32], 'y': [8, 8, 8, 8],
                              'frame': [1, 0, 0, 1], 'particle': [0, 0, 1, 1]})
        for remove in [[], [0], [1], [0, 1]]:
            actual = self.link(expected, shape=shape, remove=remove)
            assert_traj_equal(actual, expected)

    def test_two_double_overlap(self):
        shape = (24, 32)
        # (a b) c --> d    # a-c and b-c overlap
        expected = DataFrame({'x': [8, 8, 16, 24], 'y': [8, 16, 12, 12],
                              'frame': [0, 1, 0, 1], 'particle': [0, 0, 1, 1]})
        for remove in [[], [0], [1], [0, 1]]:
            actual = self.link(expected, shape=shape, remove=remove)
            assert_traj_equal(actual, expected)
        # (a b) d <-- c    # a-d and b-d overlap
        expected = DataFrame({'x': [8, 8, 16, 24], 'y': [8, 16, 12, 12],
                              'frame': [0, 1, 1, 0], 'particle': [0, 0, 1, 1]})
        for remove in [[], [0], [1], [0, 1]]:
            actual = self.link(expected, shape=shape, remove=remove)
            assert_traj_equal(actual, expected)

    def test_two_triple_overlap(self):
        shape = (24, 32)
        # a --> b
        #     c --> d    # a-c, b-c, and b-d overlap
        expected = DataFrame({'x': [8, 16, 16, 24], 'y': [8, 8, 16, 16],
                              'frame': [0, 1, 0, 1], 'particle': [0, 0, 1, 1]})
        for remove in [[], [0], [1], [0, 1]]:
            actual = self.link(expected, shape=shape, remove=remove)
            assert_traj_equal(actual, expected)
        # a --> b
        #     d <-- c    # a-d, b-d, and b-c overlap
        expected = DataFrame({'x': [8, 16, 16, 24], 'y': [8, 8, 16, 16],
                              'frame': [0, 1, 1, 0], 'particle': [0, 0, 1, 1]})
        for remove in [[], [0], [1], [0, 1]]:
            actual = self.link(expected, shape=shape, remove=remove)
            assert_traj_equal(actual, expected)
        # b <-- a
        #     c -- > d    # b-c, a-c, and a-d overlap
        expected = DataFrame({'x': [8, 16, 16, 24], 'y': [8, 8, 16, 16],
                              'frame': [1, 0, 0, 1], 'particle': [0, 0, 1, 1]})
        for remove in [[], [0], [1], [0, 1]]:
            actual = self.link(expected, shape=shape, remove=remove)
            assert_traj_equal(actual, expected)

    def test_two_full_overlap(self):
        shape = (24, 24)
        # a --> b
        # c --> d    # a-c, b-c, b-d, a-d overlap
        expected = DataFrame({'x': [8, 15, 8, 15], 'y': [8, 8, 16, 16],
                              'frame': [0, 1, 0, 1], 'particle': [0, 0, 1, 1]})
        for remove in [[], [0], [1], [0, 1]]:
            actual = self.link(expected, shape=shape, remove=remove)
            assert_traj_equal(actual, expected)

    def test_merging_subnets(self):
        shape = (24, 48)
        expected = pd.DataFrame({'x': [8, 12, 16, 20, 32, 28, 40, 36],
                                 'y': [8, 16, 8, 16, 8, 16, 8, 16],
                                 'frame': [0, 1, 0, 1, 0, 1, 0, 1],
                                 'particle': [0, 0, 1, 1, 2, 2, 3, 3]})

        for n in range(5):
            for remove in itertools.combinations(range(4), n):
                actual = self.link(expected, shape=shape, remove=remove)
                assert_traj_equal(actual, expected)

    def test_splitting_subnets(self):
        shape = (24, 48)
        expected = pd.DataFrame({'x': [8, 12, 16, 20, 32, 28, 40, 36],
                                 'y': [8, 16, 8, 16, 8, 16, 8, 16],
                                 'frame': [1, 0, 1, 0, 1, 0, 1, 0],
                                 'particle': [0, 0, 1, 1, 2, 2, 3, 3]})

        for n in range(5):
            for remove in itertools.combinations(range(4), n):
                actual = self.link(expected, shape=shape, remove=remove)
                assert_traj_equal(actual, expected)

    def test_shifting_string(self):
        shape = (24, 48)
        shift = 5
        expected = pd.DataFrame({'x': [8, 8+shift, 16, 16+shift,
                                       24, 24+shift, 32, 32+shift],
                                 'y': [8, 16, 8, 16, 8, 16, 8, 16],
                                 'frame': [0, 1, 0, 1, 0, 1, 0, 1],
                                 'particle': [0, 0, 1, 1, 2, 2, 3, 3]})

        for n in range(5):
            for remove in itertools.combinations(range(4), n):
                actual = self.link(expected, shape=shape, remove=remove)
                assert_traj_equal(actual, expected)

    def test_multiple_lost_simple(self):
        shape = (32, 32)
        #   b      a, b, c, d in frame 0
        # a e c    e in frame 1, disappears, should be linked to correct one
        #   d
        # left
        expected = pd.DataFrame({'x': [8, 16, 24, 16, 15],
                                 'y': [16, 8, 16, 24, 16],
                                 'frame': [0, 0, 0, 0, 1],
                                 'particle': [0, 1, 2, 3, 0]})
        actual = self.link(expected, shape=shape, remove=[0])
        assert_traj_equal(actual, expected)
        # top
        expected = pd.DataFrame({'x': [8, 16, 24, 16, 16],
                                 'y': [16, 8, 16, 24, 15],
                                 'frame': [0, 0, 0, 0, 1],
                                 'particle': [0, 1, 2, 3, 1]})
        actual = self.link(expected, shape=shape, remove=[0])
        assert_traj_equal(actual, expected)
        # right
        expected = pd.DataFrame({'x': [8, 16, 24, 16, 17],
                                 'y': [16, 8, 16, 24, 16],
                                 'frame': [0, 0, 0, 0, 1],
                                 'particle': [0, 1, 2, 3, 2]})
        actual = self.link(expected, shape=shape, remove=[0])
        assert_traj_equal(actual, expected)
        # bottom
        expected = pd.DataFrame({'x': [8, 16, 24, 16, 16],
                                 'y': [16, 8, 16, 24, 17],
                                 'frame': [0, 0, 0, 0, 1],
                                 'particle': [0, 1, 2, 3, 3]})
        actual = self.link(expected, shape=shape, remove=[0])
        assert_traj_equal(actual, expected)

    def test_multiple_lost_subnet(self):
        shape = (24, 48)
        #  (subnet 1, a-c) g (subnet 2, d-f)
        # left
        expected = pd.DataFrame({'x': [8, 10, 16, 32, 40, 38, 23],
                                 'y': [8, 16, 8, 8, 8, 16, 8],
                                 'frame': [0, 1, 0, 0, 0, 1, 1],
                                 'particle': [0, 0, 1, 2, 3, 3, 1]})
        for n in range(3):
            for remove in itertools.combinations(range(3), n):
                actual = self.link(expected, shape=shape, remove=remove)
                assert_traj_equal(actual, expected)
        # right
        expected = pd.DataFrame({'x': [8, 10, 16, 32, 40, 38, 25],
                                 'y': [8, 16, 8, 8, 8, 16, 8],
                                 'frame': [0, 1, 0, 0, 0, 1, 1],
                                 'particle': [0, 0, 1, 2, 3, 3, 2]})
        for n in range(3):
            for remove in itertools.combinations(range(3), n):
                actual = self.link(expected, shape=shape, remove=remove)
                assert_traj_equal(actual, expected)
