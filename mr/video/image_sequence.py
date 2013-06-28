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
    """Iterable object that returns frames of video as numpy arrays.

    Parameters
    ----------
    directory : string
    gray : Convert color image to grayscale. True by default.
    invert : Invert black and white. True by default.

    Examples
    --------
    >>> video = ImageSequence('directory_name')
    >>> imshow(video[0]) # Show the first frame.
    >>> imshow(video[1][0:10][0:10]) # Show one corner of the second frame.

    >>> for frame in video[:]:
    ...    # Do something with every frame.
 
    >>> for frame in video[10:20]:
    ...    # Do something with frames 10-20.

    >>> for frame in video[[5, 7, 13]]:
    ...    # Do something with frames 5, 7, and 13.
 
    >>> frame_count = video.count # Number of frames in video
    >>> frame_shape = video.shape # Pixel dimensions of video
    """

    def __init__(self, directory, gray=True, invert=True):
        Frames.__init__(self, directory, gray, invert)
        dummy_instance = self._open(directory)
        self.count = len(dummy_instance.files)
        self.shape = imread(dummy_instance.files[0]).shape

    def _open(self, filename):
        return open_image_sequence(filename)
