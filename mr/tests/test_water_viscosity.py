"""Reproduce a control experiment."""
import unittest
from numpy.testing import assert_almost_equal
from numpy.testing.decorators import slow
from pandas.util.testing import (assert_series_equal, assert_frame_equal)

import os

import pandas as pd
from pandas import DataFrame, Series

import mr

MPP = 100/285.
FPS = 24.

expected_n = 1.0044460473766137
expected_viscosity = 1.0710321808549723

def curpath():
    path, _ = os.path.split(os.path.abspath(__file__))
    return path

class TestWaterViscosity(unittest.TestCase):
    def setUp(self):
        DATA_DIR = 'water'
        self.path = os.path.join(curpath(), DATA_DIR)
        self.raw_traj = pd.load(os.path.join(self.path, 'raw_traj.df'))
        self.expected_drift = pd.load(os.path.join(self.path, 'drift.df'))
        self.expected_corrected_traj = pd.load(
            os.path.join(self.path, 'corrected_traj.df'))
        self.expected_imsds = pd.load(os.path.join(self.path, 'imsds.df'))
        self.expected_emsds = pd.load(os.path.join(self.path, 'emsds.df'))

    @slow
    def test_drift(self):
        self.drift = mr.compute_drift(self.raw_traj, (18, 23))
        assert_frame_equal(self.drift, self.expected_drift)
 
    @slow
    def test_drift_subtraction(self):
        corrected_traj = mr.subtract_drift(self.raw_traj,
                                           self.expected_drift)
        assert_frame_equal(corrected_traj, self.expected_corrected_traj)

    @slow
    def test_individual_msds(self):
        imsds = mr.imsd(self.expected_corrected_traj, MPP, FPS)
        assert_frame_equal(imsds, self.expected_imsds)

    @slow
    def test_ensemble_msds(self):
        emsds = mr.emsd(self.expected_corrected_traj, MPP, FPS)
        assert_frame_equal(emsds, self.expected_emsds)

    def test_fit_powerlaw(self):
        fit = mr.fit_powerlaw(
            self.expected_emsds.set_index('lagt')['msd'].dropna()).T
        viscosity = 1.740/fit['A'].values[0]
        n = fit['n'].values[0]
        assert_almost_equal(n, expected_n, decimal=3)
        assert_almost_equal(viscosity, expected_viscosity, decimal=3)

    @slow
    def test_system(self):
        drift = mr.compute_drift(self.raw_traj)
        corrected_traj = mr.subtract_drift(self.raw_traj, drift)
        emsds = mr.emsd(corrected_traj, MPP, FPS)
        fit = mr.fit_powerlaw(emsds.set_index('lagt')['msd'].dropna()).T
        viscosity = 1.740/fit['A'].values[0]
        n = fit['n'].values[0]
        assert_almost_equal(n, expected_n, decimal=3)
        assert_almost_equal(viscosity, expected_viscosity, decimal=3)

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__,'-vvs','-x','--pdb', '--pdb-failure'],
                   exit=False)

