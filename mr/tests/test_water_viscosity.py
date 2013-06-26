"""Reproduce a control experiment."""
import unittest
import nose
from numpy.testing import assert_almost_equal, assert_allclose
from numpy.testing.decorators import slow
from pandas.util.testing import (assert_series_equal, assert_frame_equal)

import os
import pandas as pd
from pandas import DataFrame, Series

import mr

DIAMETER = 11
MINMASS = 3000
MPP = 100/285.
FPS = 24.

path, _ = os.path.split(os.path.abspath(__file__))

def _skip_if_no_cv2():
    try:
        import cv2
    except ImportError:
        raise nose.SkipTest('OpenCV not installed. Skipping.')

class TestWaterViscosity(unittest.TestCase):
    def setUp(self):
        self.VIDEO_PATH = os.path.join(path, 'water/bulk-water.mov')
        STORE_PATH = os.path.join(path, 'water/expected.h5')
        self.store = pd.HDFStore(STORE_PATH, 'r')

    @slow
    def test_batch_locate_usage(self):
        _skip_if_no_cv2()
        # Only checking that it doesn't raise an error.
        frames = mr.Video(self.VIDEO_PATH)
        MAX_FRAME = 2
        features = mr.batch(frames[:MAX_FRAME], DIAMETER, MINMASS)

    @slow
    def test_track_usage(self):
        # Only checking that it doesn't raise an error
        features = self.store.select('sample_features')
        t = mr.track(features)

    def test_drift(self):
        drift = mr.motion.compute_drift(self.store['good_traj'])
        assert_frame_equal(drift, self.store['drift'])

    def test_drift_subtraction(self):
        trajectories = mr.subtract_drift(self.store['good_traj'],
                                         self.store['drift'])
        assert_frame_equal(trajectories, self.store['tm'])

    def test_individual_msds(self):
        imsds = mr.imsd(self.store['tm'], MPP, FPS)
        assert_frame_equal(imsds, self.store['individual_msds'])

    def test_ensemble_msds(self):
        detailed_em = mr.emsd(self.store['tm'], MPP, FPS, detail=True)
        print detailed_em.columns
        print self.store['ensemble_msd'].columns
        assert_frame_equal(detailed_em, self.store['ensemble_msd'])
        em = mr.emsd(self.store['tm'], MPP, FPS)
        assert_series_equal(em, self.store['ensemble_msd'].msd)

    def test_fit_powerlaw(self):
        EXPECTED_VISCOSITY = 1.01
        EXPECTED_N = 1.0
        em = self.store['ensemble_msd'].msd
        fit = mr.fit_powerlaw(em)
        A = fit['A'].values[0]
        n = fit['n'].values[0]
        viscosity = 1.740/A
        assert_allclose(n, EXPECTED_N, rtol=0.1)
        assert_allclose(viscosity, EXPECTED_VISCOSITY, rtol=0.5)

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],
                   exit=False)

