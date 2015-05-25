from numpy.testing.utils import assert_allclose
import nose
import numpy as np
from trackpy.preprocessing import *


small = np.random.randn(2**9, 2**9)
bp_scipy = bandpass(small, 3, 11)


def test_legacy_bandpass():
    lbp_numpy = legacy_bandpass(small, 3, 11)
    assert_allclose(lbp_numpy, bp_scipy, rtol=1e-3, atol=0.2)


def test_legacy_bandpass_fftw():
    try:
        import pyfftw
    except ImportError:
        raise nose.SkipTest("pyfftw not installed. Skipping.")
    lbp_fftw = legacy_bandpass_fftw(small, 3, 11)
    assert_allclose(lbp_fftw, bp_scipy)
