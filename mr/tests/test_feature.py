import nose
import unittest

import numpy as np

from mr import feature

SAMPLE_FRAME = feature.plt.imread('data/sample_frame.png')

class TestFeature(unittest.TestCase):

    def test_bandpass(self):
        result = feature.bandpass(SAMPLE_FRAME, 1, 12)
        expect = np.loadtxt('data/bandpass_result.txt')
        self.assert_(np.allclose(result, expect))

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vv', '-s', '-x'], exit=False)

