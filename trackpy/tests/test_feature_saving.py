import functools
import os
import unittest
import warnings

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import pandas

from pandas.testing import (
    assert_series_equal,
    assert_frame_equal,
)

import trackpy as tp
from trackpy.tests.common import StrictTestCase
from trackpy.tests.common import TrackpyImageSequence

# Quiet warnings about get_store being deprecated.
# These come from pandas.io and are caused by line 62:
#       s = self.storage_class(STORE_NAME)
import warnings
warnings.filterwarnings("ignore", message="get_store is deprecated")

path, _ = os.path.split(os.path.abspath(__file__))


def _random_hash():
    return ''.join(map(str, np.random.randint(0, 10, 10)))


def _skip_if_no_pytables():
    try:
        import tables
    except ImportError:
        raise unittest.SkipTest('pytables not installed. Skipping.')

    # https://github.com/soft-matter/trackpy/issues/643
    if tables.get_hdf5_version() == "1.8.5-patch1":
        raise unittest.SkipTest('this pytables version has an incompatible HDF5 version. Skipping.')

class FeatureSavingTester:
    def prepare(self, batch_params=None):
        directory = os.path.join(path, 'video', 'image_sequence')
        v = TrackpyImageSequence(os.path.join(directory, '*.png'))
        self.v = [tp.invert_image(v[i]) for i in range(2)]
        # mass depends on pixel dtype, which differs per reader
        minmass = self.v[0].max() * 2
        self.PARAMS = {'diameter': 11, 'minmass': minmass}
        if batch_params is not None:
            self.PARAMS.update(batch_params)
        self.expected = tp.batch(self.v, engine='python', meta=False,
                                 **self.PARAMS)

    def test_storage(self):
        STORE_NAME = 'temp_for_testing_{}.h5'.format(_random_hash())
        if os.path.isfile(STORE_NAME):
            os.remove(STORE_NAME)
        try:
            s = self.storage_class(STORE_NAME)
        except OSError:
            unittest.SkipTest('Cannot make an HDF5 file. Skipping')
        else:
            tp.batch(self.v, output=s, engine='python', meta=False,
                     **self.PARAMS)
            self.assertEqual(len(s), 2)
            self.assertEqual(s.max_frame, 1)
            count_total_dumped = s.dump()['frame'].nunique()
            count_one_dumped = s.dump(1)['frame'].nunique()
            self.assertEqual(count_total_dumped, 2)
            self.assertEqual(count_one_dumped, 1)
            assert_frame_equal(s.dump().reset_index(drop=True), 
                               self.expected.reset_index(drop=True))
            assert_frame_equal(s[0], s.get(0))

            # Putting an empty df should warn
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('ignore')
                warnings.simplefilter('always', UserWarning)
                s.put(pandas.DataFrame())
                assert len(w) == 1
            s.close()
            os.remove(STORE_NAME)


class TestPandasHDFStore(FeatureSavingTester, StrictTestCase):
    def setUp(self):
        _skip_if_no_pytables()
        self.prepare()
        self.storage_class = tp.PandasHDFStore


class TestPandasHDFStoreBig(FeatureSavingTester, StrictTestCase):
    def setUp(self):
        _skip_if_no_pytables()
        self.prepare()
        self.storage_class = tp.PandasHDFStoreBig

    def test_cache(self):
        """Store some frames, make a cache, then store some more frames."""
        STORE_NAME = 'temp_for_testing_{}.h5'.format(_random_hash())
        if os.path.isfile(STORE_NAME):
            os.remove(STORE_NAME)
        try:
            s = self.storage_class(STORE_NAME)
        except OSError:
            unittest.SkipTest('Cannot make an HDF5 file. Skipping')
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


class TestSingleThreaded(FeatureSavingTester, StrictTestCase):
    def setUp(self):
        _skip_if_no_pytables()

        # Check that the argument is getting passed to utils.get_pool()
        with self.assertRaises(TypeError):
            self.prepare(batch_params={'processes': 'junk'})

        self.prepare(batch_params={'processes': 1})
        self.storage_class = tp.PandasHDFStoreBig


class TestPandasHDFStoreBigCompressed(FeatureSavingTester, StrictTestCase):
    def setUp(self):
        _skip_if_no_pytables()
        self.prepare()
        self.storage_class = functools.partial(
            tp.PandasHDFStoreBig, complevel=4, complib='zlib',
            fletcher32=True)


class TestPandasHDFStoreSingleNode(FeatureSavingTester, StrictTestCase):
    def setUp(self):
        _skip_if_no_pytables()
        self.prepare()
        self.storage_class = tp.PandasHDFStoreSingleNode


class TestPandasHDFStoreSingleNodeCompressed(FeatureSavingTester,
                                             StrictTestCase):
    def setUp(self):
        _skip_if_no_pytables()
        self.prepare()
        self.storage_class = functools.partial(
            tp.PandasHDFStoreSingleNode,
            complevel=4, complib='zlib', fletcher32=True)


if __name__ == '__main__':
    import unittest
    unittest.main()
