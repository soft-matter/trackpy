import unittest
from numpy.testing.decorators import slow
import os
import pandas as pd
from mr import ptraj

path, _ = os.path.split(os.path.abspath(__file__))

class TestLabeling(unittest.TestCase):
    def setUp(self):
        self.sparse = pd.load(os.path.join(path, 'misc', 
                                           'sparse_trajectories.df'))

    @slow
    def test_labeling_sparse_trajectories(self):
        ptraj(self.sparse, label=True) # No errors?
