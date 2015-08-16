from __future__ import division
from numpy.testing.utils import assert_allclose
from trackpy.preprocessing import *
from trackpy.artificial import gen_nonoverlapping_locations, draw_spots


pos = gen_nonoverlapping_locations((512, 512), 200, 20)
frame = draw_spots((512, 512), pos, 20, noise_level=100)
margin = 11
bp_scipy = bandpass(frame, 3, 11)[margin:-margin, margin:-margin]


def test_legacy_bandpass():
    lbp_numpy = legacy_bandpass(frame, 3, 11)[margin:-margin, margin:-margin]
    assert_allclose(lbp_numpy, bp_scipy, atol=1.1)


def test_legacy_bandpass_fftw():
    try:
        import pyfftw
    except ImportError:
        raise nose.SkipTest("pyfftw not installed. Skipping.")
    lbp_fftw = legacy_bandpass_fftw(frame, 3, 11)[margin:-margin, margin:-margin]
    assert_allclose(lbp_fftw, bp_scipy, atol=1.1)

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
