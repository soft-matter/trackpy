import os
import mr
import unittest
import nose
import numpy as np
from numpy.testing import (assert_equal)

path, _ = os.path.split(os.path.abspath(__file__))
path = os.path.join(path, 'video')

def _skip_if_no_cv2():
    try:
        import cv2
    except ImportError:
        raise nose.SkipTest('OpenCV not installed. Skipping.')

class TestVideo(unittest.TestCase):

    def setUp(self):
        _skip_if_no_cv2()
        self.filename = os.path.join(path, '../water/bulk-water.mov')
        self.frame0 = np.load(os.path.join(path, 'frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'frame1.npy'))
        self.v = mr.Video(self.filename)

    def test_shape(self):
        _skip_if_no_cv2()
        assert_equal(self.v.shape, (640, 424))

    def test_count(self):
        _skip_if_no_cv2()
        assert_equal(self.v.count, 480)

    def test_iterator(self):
        _skip_if_no_cv2()
        assert_equal(self.v.next(), self.frame0)
        assert_equal(self.v.next(), self.frame1)

    def test_rewind(self):
        _skip_if_no_cv2()
        self.v.rewind()
        assert_equal(self.v.next(), self.frame0)

    def test_slicing(self):
        _skip_if_no_cv2()
        frame0, frame1 = list(self.v[0:1])
        assert_equal(frame0, self.frame0)
        assert_equal(frame1, self.frame1)
