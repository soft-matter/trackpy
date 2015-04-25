from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import unittest
import os

from numpy.testing.decorators import slow
import pandas as pd

import trackpy
from trackpy import ptraj
from trackpy.utils import suppress_plotting

# Catch attempts to set values on an inadvertent copy of a Pandas object.
trackpy.utils.make_pandas_strict()

path, _ = os.path.split(os.path.abspath(__file__))


class TestLabeling(unittest.TestCase):
    def setUp(self):
        self.sparse = pd.read_pickle(os.path.join(path, 'data', 
                                           'sparse_trajectories.df'))

    @slow
    def test_labeling_sparse_trajectories(self):
        suppress_plotting()
        ptraj(self.sparse, label=True) # No errors?
if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
