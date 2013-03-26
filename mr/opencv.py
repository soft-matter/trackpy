import cv
from PIL import Image
import numpy as np

def open_video(filename):
    """Thin convenience function for return an opencv Capture object.
    Pass the result to frame_generator() to get images."""
    # ffmpeg -i unreadable.avi -sameq -r 30 readable.avi
    capture = cv.CaptureFromFile(filename)
    return capture

def frame_generator(capture, gray=True, invert=True):
    """Return a generator that yields frames of video as image arrays.

    Parameters
    ----------
    capture: an opencv Capture object (See mr.open_video.)
    gray: Convert frames to grayscale. True by default.
    invert: Invert black and white. True by default.

    Returns
    -------
    a generator object that yields a frame on each iteration until it reaches
    the end of the captured video
    """
    count = int(cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FRAME_COUNT))
    for i in range(count):
        frame = cv.QueryFrame(capture)
        if frame is None:
            # Frame count is not always accurate.
            return
        if gray:
            size = frame.width, frame.height
            gray_frame = cv.CreateImage(size, frame.depth, 1)
            cv.CvtColor(frame, gray_frame, cv.CV_RGB2GRAY)
            frame = gray_frame
        a = np.asarray(cv.GetMat(frame))
        if invert:
            invert_in_place(a)
        yield a 

def invert_in_place(a):
    a *= -1
    a += 255
