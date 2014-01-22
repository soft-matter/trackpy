import unittest
import nose

from numpy.testing import assert_almost_equal, assert_allclose
from numpy.testing.decorators import slow
from pandas.util.testing import (assert_series_equal, assert_frame_equal)

import os
from tempfile import NamedTemporaryFile
import pandas as pd
from pandas import DataFrame, Series

import trackpy as tp 
import sqlite3

path, _ = os.path.split(os.path.abspath(__file__))

class TestFeatureSaving(unittest.TestCase):

    def setUp(self):
        self.db_conn = sqlite3.connect(':memory:')
        directory = os.path.join(path, 'video', 'image_sequence')
        self.v = tp.ImageSequence(directory)
        self.PARAMS = (11, 3000)
        with NamedTemporaryFile() as temp:
            self.expected = tp.batch(self.v[[0, 1]], *self.PARAMS,
                                     meta=temp.name)

    def test_sqlite(self):
        with NamedTemporaryFile() as temp:
            f = tp.batch(self.v[[0, 1]], *self.PARAMS, conn=self.db_conn,
                     sql_flavor='sqlite', table='features', meta=temp.name)
        assert_frame_equal(f, self.expected)

    def test_HDFStore(self):
        STORE_NAME = 'temp_for_testing.h5'
        if os.path.isfile(STORE_NAME):
            os.remove(STORE_NAME)
        try:
            store = pd.HDFStore(STORE_NAME)
        except:
            nose.SkipTest('Cannot make an HDF5 file. Skipping')
        else:
            with NamedTemporaryFile() as temp:
                f = tp.batch(self.v[[0, 1]], *self.PARAMS, store=store,
                             table='features', meta=temp.name)
            assert_frame_equal(f.reset_index(drop=True), 
                           self.expected.reset_index(drop=True))
            os.remove(STORE_NAME)
