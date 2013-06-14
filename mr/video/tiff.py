import os
import cv2
import numpy as np
from libtiff import TIFF
from mr.video.opencv import Video

class PseudoCapture(object):
    def __init__(self, filename):
        self.filename = filename
        self.tiff = TIFF.open(filename)
        self.count = self._count_frames()
        self.end = False
        self.generator = self.tiff.iter_images()
    def read(self):
        try:
            return True, self.generator.next()
        except StopIteration:
            return False, np.array([])
    def _count_frames(self):
        return sum([1 for _ in TIFF.open(self.filename).iter_images()])
    def get(self, code):
        if code == cv2.cv.CV_CAP_PROP_FRAME_WIDTH:
            return self.tiff.read_image().shape[0]
        if code == cv2.cv.CV_CAP_PROP_FRAME_HEIGHT:
            return self.tiff.read_image().shape[1]
        if code == cv2.cv.CV_CAP_PROP_FRAME_COUNT:
            return self.count 

def open_tiffstack(filename):
    """Thin convenience function for return an opencv2 Capture object."""
    # ffmpeg -i unreadable.avi -sameq -r 30 readable.avi
    if not os.path.isfile(filename):
        raise ValueError, "%s is not a file." % filename
    capture = PseudoCapture(filename)
    return capture

class TiffStack(Video):
    def _open(self, filename):
        return open_tiffstack(filename)
