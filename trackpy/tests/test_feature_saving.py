from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import functools
import nose
import warnings
import os

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import pandas
from pandas.util.testing import (assert_series_equal, assert_frame_equal)

import trackpy as tp
from trackpy.tests.common import StrictTestCase
from trackpy.tests.common import TrackpyImageSequence

# Quiet warnings about get_store being deprecated.
# These come from pandas.io and are caused by line 62:
#       s = self.storage_class(STORE_NAME)
import warnings
warnings.filterwarnings("ignore", message="get_store is deprecated")

path, _ = os.path.split(os.path.abspath(__file__))


# This is six stuff here because pandas.HDFStore is fussy about the string type of one of
# its option args. There seems to be no good reason for that at all.
if six.PY2:
    zlib = six.binary_type('zlib')
elif six.PY3:
    zlib = 'zlib'
else:
    raise("six is confused")


def _random_hash():
    return ''.join(map(str, np.random.randint(0, 10, 10)))


def _skip_if_no_pytables():
    try:
        import tables
    except ImportError:
        raise nose.SkipTest('pytables not installed. Skipping.')


class FeatureSavingTester(object):
    def prepare(self):
        directory = os.path.join(path, 'video', 'image_sequence')
        v = TrackpyImageSequence(os.path.join(directory, '*.png'))
        self.v = [tp.invert_image(v[i]) for i in range(2)]
        # mass depends on pixel dtype, which differs per reader
        minmass = self.v[0].max() * 2
        self.PARAMS = {'diameter': 11, 'minmass': minmass}
        self.expected = tp.batch(self.v, engine='python', meta=False,
                                 **self.PARAMS)

    def test_storage(self):
        STORE_NAME = 'temp_for_testing_{0}.h5'.format(_random_hash())
        if os.path.isfile(STORE_NAME):
            os.remove(STORE_NAME)
        try:
            s = self.storage_class(STORE_NAME)
        except IOError:
            nose.SkipTest('Cannot make an HDF5 file. Skipping')
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
        STORE_NAME = 'temp_for_testing_{0}.h5'.format(_random_hash())
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


class TestPandasHDFStoreBigCompressed(FeatureSavingTester, StrictTestCase):
    def setUp(self):
        _skip_if_no_pytables()
        self.prepare()
        self.storage_class = functools.partial(
            tp.PandasHDFStoreBig, complevel=4, complib=zlib,
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
            complevel=4, complib=zlib, fletcher32=True)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
