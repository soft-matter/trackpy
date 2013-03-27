import os
import cv2
from PIL import Image
import numpy as np

def open_video(filename):
    """Thin convenience function for return an opencv2 Capture object.
    Pass the result to frame_generator() to get images."""
    # ffmpeg -i unreadable.avi -sameq -r 30 readable.avi
    if not os.path.isfile(filename):
        raise ValueError, "%s is not a file." % filename
    capture = cv2.VideoCapture(filename)
    return capture

def frame_generator(filename, start_frame=0,
                    gray=True, invert=True):
    """Return a generator that yields frames of video as image arrays.

    Parameters
    ----------
    filename: path to video file
    start_frame: Fast-forward to frame number
    gray: Convert frames to grayscale. True by default.
    invert: Invert black and white. True by default.

    Returns
    -------
    a generator object that yields a frame on each iteration until it reaches
    the end of the captured video
    """
    if not os.path.isfile(filename):
        raise ValueError, "%s is not a file." % filename
    capture = cv2.VideoCapture(filename)
    count = int(capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    if start_frame > 0:
        print "Seeking through video to starting frame..."
        [capture.read()[0] for _ in range(start_frame)] # seek
    for i in range(count - start_frame):
        ret, frame = capture.read()
        if not ret:
            # A failsafe: the frame count is not always accurate.
            return
        if i < start_frame:
            continue # seek without yielding frames
        if gray:
            frame = cv2.cvtColor(frame, cv2.cv.CV_RGB2GRAY)
        if invert:
            invert_in_place(frame)
        yield frame 

def invert_in_place(a):
    a *= -1
    a += 255
