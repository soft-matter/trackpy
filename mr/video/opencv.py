import os
import numpy as np
import cv2
from mr.video.frames import Frames

def open_video(filename):
    """Thin convenience function for return an opencv2 Capture object."""
    # ffmpeg -i unreadable.avi -sameq -r 30 readable.avi
    if not os.path.isfile(filename):
        raise ValueError, "%s is not a file." % filename
    capture = cv2.VideoCapture(filename)
    return capture

class Video(Frames):
    """Iterable object that returns frames of video as numpy arrays of integers
    0-255.

    Parameters
    ----------
    filename : string
    gray : Convert color image to grayscale. True by default.
    invert : Invert black and white. True by default.

    Examples
    --------
    >>> video = Video('filename')
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
        self.shape = (int(self.capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
                      int(self.capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
        self.count = int(self.capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    def _open(self, filename):
        return open_video(filename)

    def _process(self, frame):
        if self.gray:
            frame = cv2.cvtColor(frame, cv2.cv.CV_RGB2GRAY)
        if self.invert:
            frame *= -1
            frame += 255
        return frame 
