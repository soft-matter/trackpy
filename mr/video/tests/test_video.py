import os
import mr
import unittest
import nose
import numpy as np
from numpy.testing import (assert_equal)

path, _ = os.path.split(os.path.abspath(__file__))
path = os.path.join(path, 'data')

def _skip_if_no_cv2():
    try:
        import cv2
    except ImportError:
        raise nose.SkipTest('OpenCV not installed. Skipping.')

def _skip_if_no_libtiff():
    try:
        import libtiff
    except ImportError:
        raise nose.SkipTest('libtiff not installed. Skipping.')

class TestVideo(unittest.TestCase):

    def setUp(self):
        _skip_if_no_cv2()
        self.filename = os.path.join(path, 'bulk-water.mov')
        self.frame0 = np.load(os.path.join(path, 'bulk-water_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'bulk-water_frame1.npy'))
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

    def test_getting_slice(self):
        _skip_if_no_cv2()
        frame0, frame1 = list(self.v[0:1])
        assert_equal(frame0, self.frame0)
        assert_equal(frame1, self.frame1)

    def test_getting_single_frame(self):
        _skip_if_no_cv2()
        assert_equal(self.v[1], self.frame1)
        assert_equal(self.v[0], self.frame0)
        assert_equal(self.v[0], self.frame0)
        assert_equal(self.v[1], self.frame1)
        assert_equal(self.v[1], self.frame1)

    def test_getting_list(self):
        _skip_if_no_cv2()
        actual = list(self.v[[1, 0, 0, 1, 1]])
        expected = [self.frame1, self.frame0, self.frame0, self.frame1,
                    self.frame1]
        [assert_equal(a, b) for a, b in zip(actual, expected)]

    def test_invert(self):
        inverted_frame0 = self.frame0 ^ np.iinfo(self.frame0.dtype).max
        inverted_video = mr.Video(self.filename, invert=False)
        assert_equal(inverted_video[0], inverted_frame0)

    def test_integer_attributes(self):
        assert_equal(len(self.v.shape), 2)
        self.assertTrue(isinstance(self.v.shape[0], int))
        self.assertTrue(isinstance(self.v.shape[1], int))
        self.assertTrue(isinstance(self.v.count, int))


class TestTiffStack(unittest.TestCase):

    def setUp(self):
        _skip_if_no_libtiff()
        self.filename = os.path.join(path, 'stuck.tif')
        self.frame0 = np.load(os.path.join(path, 'stuck_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'stuck_frame1.npy'))
        self.v = mr.TiffStack(self.filename, invert=False)

    def test_shape(self):
        _skip_if_no_libtiff()
        assert_equal(self.v.shape, (512, 512))

    def test_count(self):
        _skip_if_no_libtiff()
        assert_equal(self.v.count, 5)

    def test_iterator(self):
        _skip_if_no_libtiff()
        assert_equal(self.v.next(), self.frame0)
        assert_equal(self.v.next(), self.frame1)

    def test_rewind(self):
        _skip_if_no_libtiff()
        self.v.rewind()
        assert_equal(self.v.next(), self.frame0)

    def test_getting_slice(self):
        _skip_if_no_libtiff()
        frame0, frame1 = list(self.v[0:1])
        assert_equal(frame0, self.frame0)
        assert_equal(frame1, self.frame1)

    def test_getting_single_frame(self):
        _skip_if_no_libtiff()
        assert_equal(self.v[1], self.frame1)
        assert_equal(self.v[0], self.frame0)
        assert_equal(self.v[0], self.frame0)
        assert_equal(self.v[1], self.frame1)
        assert_equal(self.v[1], self.frame1)

    def test_getting_list(self):
        _skip_if_no_libtiff()
        actual = list(self.v[[1, 0, 0, 1, 1]])
        expected = [self.frame1, self.frame0, self.frame0, self.frame1,
                    self.frame1]
        [assert_equal(a, b) for a, b in zip(actual, expected)]

    def test_invert(self):
        inverted_frame0 = self.frame0 ^ np.iinfo(self.frame0.dtype).max
        inverted_tiff_stack = mr.TiffStack(self.filename, invert=True)
        assert_equal(inverted_tiff_stack[0], inverted_frame0)

    def test_integer_attributes(self):
        assert_equal(len(self.v.shape), 2)
        self.assertTrue(isinstance(self.v.shape[0], int))
        self.assertTrue(isinstance(self.v.shape[1], int))
        self.assertTrue(isinstance(self.v.count, int))


class TestImageSequence(unittest.TestCase):

    def setUp(self):
        self.filename = os.path.join(path, 'image_sequence')
        self.frame0 = np.load(os.path.join(path, 'seq_frame0.npy'))
        self.frame1 = np.load(os.path.join(path, 'seq_frame1.npy'))
        self.v = mr.ImageSequence(self.filename, invert=False)

    def test_shape(self):
        assert_equal(self.v.shape, (424, 640))

    def test_count(self):
        assert_equal(self.v.count, 5)

    def test_iterator(self):
        assert_equal(self.v.next(), self.frame0)
        assert_equal(self.v.next(), self.frame1)

    def test_rewind(self):
        self.v.rewind()
        assert_equal(self.v.next(), self.frame0)

    def test_getting_slice(self):
        frame0, frame1 = list(self.v[0:1])
        assert_equal(frame0, self.frame0)
        assert_equal(frame1, self.frame1)

    def test_getting_single_frame(self):
        assert_equal(self.v[0], self.frame0)
        assert_equal(self.v[0], self.frame0)
        assert_equal(self.v[1], self.frame1)
        assert_equal(self.v[1], self.frame1)

    def test_getting_list(self):
        actual = list(self.v[[1, 0, 0, 1, 1]])
        expected = [self.frame1, self.frame0, self.frame0, self.frame1,
                    self.frame1]
        [assert_equal(a, b) for a, b in zip(actual, expected)]

    def test_invert(self):
        inverted_frame0 = self.frame0 ^ np.iinfo(self.frame0.dtype).max
        inverted_sequence = mr.ImageSequence(self.filename, invert=True)
        assert_equal(inverted_sequence[0], inverted_frame0)

    def test_integer_attributes(self):
        assert_equal(len(self.v.shape), 2)
        self.assertTrue(isinstance(self.v.shape[0], int))
        self.assertTrue(isinstance(self.v.shape[1], int))
        self.assertTrue(isinstance(self.v.count, int))
