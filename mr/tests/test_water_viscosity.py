"""Reproduce a control experiment."""
import unittest
from numpy.testing import assert_almost_equal
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

def curpath():
    path, _ = os.path.split(os.path.abspath(__file__))
    return path

class TestWaterViscosity(unittest.TestCase):
    def setUp(self):
        VIDEO_PATH = os.path.join(curpath(), 'water/bulk-water.mov')
        STORE_PATH = os.path.join(curpath(), 'water/expected.h5')
        self.store = pd.HDFStore(STORE_PATH)
        self.frames = mr.video.frame_generator(VIDEO_PATH)

    @slow
    def test_batch_locate(self):
        features = mr.feature.batch(self.frames, DIAMETER, MINMASS)
        assert_frame_equal(features, self.store['features'])

    @slow
    def test_track(self):
        features = self.store['features']
        trajectories = mr.tracking.track(features[features['ecc'] < 0.1])
        trajectories = mr.tracking.bust_ghosts(trajectories, 50)
        assert_frame_equal(trajectories, self.store['raw_trajectories'])

    def test_drift(self):
        drift = mr.motion.compute_drift(self.store['raw_trajectories'])
        assert_frame_equal(drift, self.store['drift'])

    def test_drift_subtraction(self):
        trajectories = mr.motion.subtract_drift(self.store['raw_trajectories'],
                                                self.store['drift'])
        assert_frame_equal(trajectories, self.store['trajectories'])

    def test_individual_msds(self):
        imsds = mr.imsd(self.store['trajectories'], MPP, FPS)
        assert_frame_equal(imsds, self.store['individual_msds'])

    def test_ensemble_msds(self):
        em = mr.emsd(self.store['trajectories'], MPP, FPS)
        assert_frame_equal(em, self.store['ensemble_msd'])

    def test_fit_powerlaw(self):
        EXPECTED_VISCOSITY = 1.01
        EXPECTED_N = 1.0
        em = self.store['ensemble_msd']
        fit = mr.fit_powerlaw(em.set_index('lagt')['msd'].dropna(), plot=False)
        A = fit['A'].values[0]
        n = fit['n'].values[0]
        viscosity = 1.740/A
        assert_almost_equal(n, EXPECTED_N, decimal=1)
        assert_almost_equal(viscosity, EXPECTED_VISCOSITY, decimal=1)

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],
                   exit=False)

