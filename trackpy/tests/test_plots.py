
import unittest
from numpy.testing.decorators import slow
import os

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from trackpy import plots
from trackpy.utils import suppress_plotting

path, _ = os.path.split(os.path.abspath(__file__))

class TestPlots(unittest.TestCase):
    def setUp(self):
        self.sparse = pd.read_pickle(os.path.join(path, 'data', 
                                           'sparse_trajectories.df'))

    @slow
    def test_labeling_sparse_trajectories(self):
        suppress_plotting()
        plots.plot_traj(self.sparse, label=True)

    def test_annotate(self):
        suppress_plotting()
        f = DataFrame({'x': [0, 1], 'y': [0, 1], 'frame': [0, 0],
                      'mass': [10, 20]})
        frame = np.random.randint(0, 255, (5, 5))

        # Basic usage
        plots.annotate(f, frame)
        plots.annotate(f, frame, color='r')

        # Coloring by threshold
        plots.annotate(f, frame, split_category='mass',
                       split_thresh=15, color=['r', 'g'])
        plots.annotate(f, frame, split_category='mass',
                       split_thresh=[15], color=['r', 'g'])
        plots.annotate(f, frame, split_category='mass',
                       split_thresh=[15, 25], color=['r', 'g', 'b'])

        # Check that bad parameters raise an error.

        # Too many colors
        bad_call = lambda: plots.annotate(
            f, frame, split_category='mass', split_thresh=15, color=['r', 'g', 'b'])
        self.assertRaises(bad_call)

        # Not enough colors
        bad_call = lambda: plots.annotate(
            f, frame, split_category='mass', split_thresh=15, color=['r'])
        self.assertRaises(bad_call)
        bad_call = lambda: plots.annotate(
            f, frame, split_category='mass', split_thresh=15, color='r')
        self.assertRaises(bad_call)

        # Nonexistent column name for split_category
        bad_call = lambda: plots.annotate(
            f, frame, split_category='not a column', split_thresh=15, color='r')
        self.assertRaises(bad_call)


if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
