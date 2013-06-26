import os
import numpy as np
from scipy.ndimage import imread
from mr.video.frames import Frames

class PseudoCapture(object):
    def __init__(self, directory):
        self.filename = directory # used by Frames 
        self.files = [os.path.join(directory, f) \
                      for f in os.listdir(directory)]
        self.files.sort()
        self.end = False
        self.generator = (imread(f) for f in self.files)
    def read(self):
        try:
            return True, self.generator.next()
        except StopIteration:
            return False, np.array([])

def open_image_sequence(directory):
    if not os.path.isdir(directory):
        raise ValueError, "%s is not a directory." % directory
    capture = PseudoCapture(directory)
    return capture

class ImageSequence(Frames):

    def __init__(self, directory, gray=False, invert=False):
        Frames.__init__(self, directory, gray, invert)
        dummy_instance = self._open(directory)
        self.count = len(dummy_instance.files)
        self.shape = imread(dummy_instance.files[0]).shape

    def _open(self, filename):
        return open_image_sequence(filename)

    def _process(self, frame):
        if self.gray:
            raise NotImplementedError, \
                "You must convert to the images to grayscale on your own."
        if self.invert:
            raise NotImplementedError, \
                "You must convert to the images to grayscale on your own."
        return frame

