import unittest
import nose

import functools

from numpy.testing import assert_almost_equal, assert_allclose
from numpy.testing.decorators import slow
from pandas.util.testing import (assert_series_equal, assert_frame_equal)

import os

import trackpy as tp 

path, _ = os.path.split(os.path.abspath(__file__))

def _skip_if_no_pytables():
    try:
        import tables
    except ImportError:
        raise nose.SkipTest('pytables not installed. Skipping.')


class FeatureSavingTester(object):

    def prepare(self):
        directory = os.path.join(path, 'video', 'image_sequence')
        self.v = tp.ImageSequence(directory)
        self.PARAMS = (11, 3000)
        self.expected = tp.batch(self.v[[0, 1]], *self.PARAMS,
                                 engine='python', meta=False)

    def test_storage(self):
        STORE_NAME = 'temp_for_testing.h5'
        if os.path.isfile(STORE_NAME):
            os.remove(STORE_NAME)
        try:
            s = self.storage_class(STORE_NAME)
        except IOError:
            nose.SkipTest('Cannot make an HDF5 file. Skipping')
        else:
            tp.batch(self.v[[0, 1]], *self.PARAMS,
                     output=s, engine='python', meta=False)
            print s.store.keys()
            print dir(s.store.root)
            assert len(s) == 2
            assert s.max_frame == 1
            assert_frame_equal(s.dump().reset_index(drop=True), 
                               self.expected.reset_index(drop=True))
            assert_frame_equal(s[0], s.get(0))
            s.close()
            os.remove(STORE_NAME)


class TestPandasHDFStore(FeatureSavingTester, unittest.TestCase):
    def setUp(self):
        _skip_if_no_pytables()
        self.prepare()
        self.storage_class = tp.PandasHDFStore


class TestPandasHDFStoreBig(FeatureSavingTester, unittest.TestCase):
    def setUp(self):
        _skip_if_no_pytables()
        self.prepare()
        self.storage_class = tp.PandasHDFStoreBig

    def test_cache(self):
        """Store some frames, make a cache, then store some more frames."""
        STORE_NAME = 'temp_for_testing.h5'
        if os.path.isfile(STORE_NAME):
            os.remove(STORE_NAME)
        try:
            s = self.storage_class(STORE_NAME)
        except IOError:
            nose.SkipTest('Cannot make an HDF5 file. Skipping')
        else:
            framedata = self.expected[self.expected.frame == 0]
            def putfake(store, i):
                fdat = framedata.copy()
                fdat.frame = i
                store.put(fdat)
            for i in range(10): putfake(s, i)
            assert s._frames_cache is None
            s._flush_cache() # Should do nothing
            assert set(range(10)) == set(s.frames) # Make cache
            assert set(range(10)) == set(s.frames) # Hit memory cache
            assert s._frames_cache is not None
            assert s._cache_dirty
            assert s._CACHE_NAME not in s.store

            s._flush_cache()
            assert s._CACHE_NAME in s.store
            assert not s._cache_dirty

            # Invalidate cache
            for i in range(10, 20): putfake(s, i)
            assert s._frames_cache is None
            assert s._CACHE_NAME not in s.store
            assert set(range(20)) == set(s.frames)
            assert s._frames_cache is not None

            s.rebuild_cache() # Just to try it

            s.close() # Write cache

            # Load cache from disk
            s = self.storage_class(STORE_NAME, 'r')
            assert set(range(20)) == set(s.frames) # Hit cache
            assert not s._cache_dirty

            s.close()
            os.remove(STORE_NAME)


class TestPandasHDFStoreBigCompressed(FeatureSavingTester, unittest.TestCase):
    def setUp(self):
        _skip_if_no_pytables()
        self.prepare()
        self.storage_class = functools.partial(
            tp.PandasHDFStoreBig, complevel=4, complib='zlib', fletcher32=True)


class TestPandasHDFStoreSingleNode(FeatureSavingTester, unittest.TestCase):
    def setUp(self):
        _skip_if_no_pytables()
        self.prepare()
        self.storage_class = tp.PandasHDFStoreSingleNode


class TestPandasHDFStoreSingleNodeCompressed(FeatureSavingTester,
                                             unittest.TestCase):
    def setUp(self):
        _skip_if_no_pytables()
        self.prepare()
        self.storage_class = functools.partial(
            tp.PandasHDFStoreSingleNode,
            complevel=4, complib='zlib', fletcher32=True)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
