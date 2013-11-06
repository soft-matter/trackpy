import unittest
from numpy.testing.decorators import slow
import os
import pandas as pd
from mr import ptraj
from mr.utils import suppress_plotting

path, _ = os.path.split(os.path.abspath(__file__))

class TestLabeling(unittest.TestCase):
    def setUp(self):
        self.sparse = pd.read_pickle(os.path.join(path, 'data', 
                                           'sparse_trajectories.df'))

    @slow
    def test_labeling_sparse_trajectories(self):
        suppress_plotting()
        ptraj(self.sparse, label=True) # No errors?
