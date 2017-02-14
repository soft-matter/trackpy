from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import os

from numpy.testing.decorators import slow
import pandas as pd

from trackpy import ptraj
from trackpy.utils import suppress_plotting
from trackpy.tests.common import StrictTestCase

path, _ = os.path.split(os.path.abspath(__file__))


class TestLabeling(StrictTestCase):
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
