import nose
import unittest

from mr import feature

class TestFeature(unittest.TestCase):

    SAMPLE_IMAGE = 'data/sample_frame.png'

    def test_bandpass(self):
        
