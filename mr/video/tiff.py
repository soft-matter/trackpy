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
    """Iterable object that returns frames of video as numpy arrays.

    Parameters
    ----------
    filename : string
    gray : Convert color image to grayscale. True by default.
    invert : Invert black and white. True by default.

    Examples
    --------
    >>> video = TiffStack('filename')
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
    def __init__(self, filename, gray=True, invert=True):
        Frames.__init__(self, filename, gray, invert)
        dummy_instance = self._open(filename)
        self.count = dummy_instance._count
        self.shape = dummy_instance._shape

    def _open(self, filename):
        return open_tiffstack(filename)
