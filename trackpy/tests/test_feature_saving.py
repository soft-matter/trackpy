import unittest
import nose

from numpy.testing import assert_almost_equal, assert_allclose
from numpy.testing.decorators import slow
from pandas.util.testing import (assert_series_equal, assert_frame_equal)

import os

import trackpy as tp 

path, _ = os.path.split(os.path.abspath(__file__))

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
        self.prepare()
        self.storage_class = tp.PandasHDFStore


class TestPandasHDFStoreSingleNode(FeatureSavingTester, unittest.TestCase):
    def setUp(self):
        self.prepare()
        self.storage_class = tp.PandasHDFStoreSingleNode

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
