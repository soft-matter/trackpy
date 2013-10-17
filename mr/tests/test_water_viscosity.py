"""Reproduce a control experiment."""
import unittest
import nose
from numpy.testing import assert_almost_equal, assert_allclose
from numpy.testing.decorators import slow
from pandas.util.testing import (assert_series_equal, assert_frame_equal)
from mr.core.utils import suppress_plotting

import os
import pandas as pd
from pandas import DataFrame, Series

import mr

MAX_FRAME = 300
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

    @slow
    def test_water_viscosity(self):
        _skip_if_no_cv2()
        # Only checking that it doesn't raise an error.
        frames = mr.Video(self.VIDEO_PATH)
        f = mr.batch(frames[:MAX_FRAME], DIAMETER, MINMASS)

        t = mr.track(features, 5)
        t1 = mr.bust_ghosts(t, 20)
        d = mr.drift(t1, 5)
        tm = mr.subtract_drift(t1, d)
        em = mr.emsd(tm, MPP, FPS)
        suppress_plotting()
        EXPECTED_VISCOSITY = 1.01
        EXPECTED_N = 1.0
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

