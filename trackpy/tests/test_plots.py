import os
import unittest

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import trackpy
from trackpy import plots
from trackpy.utils import suppress_plotting, fit_powerlaw
from trackpy.tests.common import StrictTestCase

# Quiet warnings about Axes not being compatible with tight_layout
import warnings
warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")

path, _ = os.path.split(os.path.abspath(__file__))

try:
    import pims
except ImportError:
    PIMS_AVAILABLE = False
else:
    PIMS_AVAILABLE = True


def _skip_if_no_pims():
    if not PIMS_AVAILABLE:
        raise unittest.SkipTest('PIMS not installed. Skipping.')


class TestPlots(StrictTestCase):
    def setUp(self):
        # older matplotlib may raise an invalid error
        np.seterr(invalid='ignore')
        self.sparse = pd.DataFrame(np.load(
            os.path.join(path, 'data', 'sparse_trajectories.npy')))

    def test_labeling_sparse_trajectories(self):
        suppress_plotting()
        plots.plot_traj(self.sparse, label=True)

    def test_ptraj_empty(self):
        suppress_plotting()
        f = lambda: plots.plot_traj(DataFrame(columns=self.sparse.columns))
        self.assertRaises(ValueError, f)

    def test_ptraj_unicode_labels(self):
        # smoke test
        plots.plot_traj(self.sparse, mpp=0.5)

    def test_ptraj_t_column(self):
        suppress_plotting()
        df = self.sparse.copy()
        cols = list(df.columns)
        cols[cols.index('frame')] = 'arbitrary name'
        df.columns = cols
        plots.plot_traj(df, t_column='arbitrary name')

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
        self.assertRaises(ValueError, bad_call)

        # Not enough colors
        bad_call = lambda: plots.annotate(
            f, frame, split_category='mass', split_thresh=15, color=['r'])
        self.assertRaises(ValueError, bad_call)
        bad_call = lambda: plots.annotate(
            f, frame, split_category='mass', split_thresh=15, color='r')
        self.assertRaises(ValueError, bad_call)

        # Nonexistent column name for split_category
        bad_call = lambda: plots.annotate(
            f, frame, split_category='not a column', split_thresh=15, color='r')
        self.assertRaises(ValueError, bad_call)

        # 3D image
        bad_call = lambda: plots.annotate(f, frame[np.newaxis, :, :])
        self.assertRaises(ValueError, bad_call)

    def test_annotate3d(self):
        _skip_if_no_pims()
        suppress_plotting()
        f = DataFrame({'x': [0, 1], 'y': [0, 1], 'z': [0, 1], 'frame': [0, 0],
                      'mass': [10, 20]})
        frame = np.random.randint(0, 255, (5, 5, 5))

        plots.annotate3d(f, frame)
        plots.annotate3d(f, frame, color='r')

        # 2D image
        bad_call = lambda: plots.annotate3d(f, frame[0])
        self.assertRaises(ValueError, bad_call)

        # Rest of the functionality is covered by annotate tests

    def test_fit_powerlaw(self):
        # smoke test
        suppress_plotting()
        em = Series([1, 2, 3], index=[1, 2, 3])
        fit_powerlaw(em)
        fit_powerlaw(em, plot=False)


if __name__ == '__main__':
    import unittest
    unittest.main()
