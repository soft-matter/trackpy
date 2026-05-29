import unittest

from numpy.testing import assert_allclose

from trackpy.preprocessing import (bandpass, legacy_bandpass,
                                   legacy_bandpass_fftw)
from trackpy.artificial import gen_nonoverlapping_locations, draw_spots
from trackpy.tests.common import StrictTestCase


class LegacyPreprocessingTests(StrictTestCase):
    def setUp(self):
        pos = gen_nonoverlapping_locations((512, 512), 200, 20)
        self.frame = draw_spots((512, 512), pos, 20, noise_level=100)
        self.margin = 11
        self.bp_scipy = bandpass(self.frame, 2, 11)[self.margin:-self.margin,
                                                    self.margin:-self.margin]

    def test_legacy_bandpass(self):
        lbp_numpy = legacy_bandpass(self.frame, 2, 5)[self.margin:-self.margin,
                                                      self.margin:-self.margin]
        assert_allclose(lbp_numpy, self.bp_scipy, atol=1.1)

    def test_legacy_bandpass_fftw(self):
        try:
            import pyfftw
        except ImportError:
            raise unittest.SkipTest("pyfftw not installed. Skipping.")
        lbp_fftw = legacy_bandpass_fftw(self.frame, 2, 5)[self.margin:-self.margin,
                                                          self.margin:-self.margin]
        assert_allclose(lbp_fftw, self.bp_scipy, atol=1.1)


if __name__ == '__main__':
    import unittest
    unittest.main()
