"""Reproduce a control experiment."""
import unittest
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

class TestWaterViscosity(unittest.TestCase):
    def setUp(self):
        VIDEO_PATH = os.path.join(path, 'water/bulk-water.mov')
        STORE_PATH = os.path.join(path, 'water/expected2.h5')
        self.store = pd.HDFStore(STORE_PATH, 'r')
        self.frames = mr.Video(VIDEO_PATH)

    @slow
    def test_batch_locate(self):
        MAX_FRAME = 2
        temp_store = pd.HDFStore('temp_for_testing.h5')
        try:
            features = mr.batch(
                temp_store, self.frames[:MAX_FRAME], DIAMETER, MINMASS)
        finally:
            os.remove('temp_for_testing.h5') 
        expected = self.store.select('all_features', 'frame<=%s' % MAX_FRAME)
        print features.frame.max(), expected.frame.max()
        assert_frame_equal(features, expected)

    @slow
    def test_track(self):
        features = self.store.select('sample_features')
        t = mr.track(features)
        expected = self.store.select('sample_traj')
        assert_frame_equal(t, expected)

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
        em = mr.emsd(self.store['tm'], MPP, FPS)
        assert_frame_equal(em, self.store['ensemble_msd'])

    def test_fit_powerlaw(self):
        EXPECTED_VISCOSITY = 1.01
        EXPECTED_N = 1.0
        em = self.store['ensemble_msd']
        fit = mr.fit_powerlaw(em.set_index('lagt')['msd'].dropna(), plot=False)
        A = fit['A'].values[0]
        n = fit['n'].values[0]
        viscosity = 1.740/A
        assert_allclose(n, EXPECTED_N, rtol=0.05)
        assert_allclose(viscosity, EXPECTED_VISCOSITY, rtol=0.15)

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],
                   exit=False)

