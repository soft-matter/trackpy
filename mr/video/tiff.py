import os
from libtiff import TIFF
import PIL.ImageOps
import numpy as np
from mr.video.frames import Frames

class PseudoCapture(object):
    def __init__(self, filename):
        self.filename = filename
        self.tiff = TIFF.open(filename)
        self._count = self._count_frames() # used once by TiffStack
        self._shape = self.tiff.read_image().shape # used once by TiffStack
        self.end = False
        self.generator = self.tiff.iter_images()
    def read(self):
        try:
            return True, self.generator.next()
        except StopIteration:
            return False, np.array([])
    def _count_frames(self):
        return len([1 for _ in TIFF.open(self.filename).iter_images()])

def open_tiffstack(filename):
    if not os.path.isfile(filename):
        raise ValueError, "%s is not a file." % filename
    capture = PseudoCapture(filename)
    return capture

class TiffStack(Frames):

    def __init__(self, filename, gray=False, invert=False):
        Frames.__init__(self, filename, gray, invert)
        dummy_instance = self._open(filename)
        self.count = dummy_instance._count
        self.shape = dummy_instance._shape

    def _open(self, filename):
        return open_tiffstack(filename)

    def _process(self, frame):
        if self.gray:
            raise NotImplementedError, \
                "You must convert to the images to grayscale on your own."
        if self.invert:
            raise NotImplementedError, \
                "You must convert to the images to grayscale on your own."
        return frame

