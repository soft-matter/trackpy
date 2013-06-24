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

    def __init__(self, filename, gray=True, invert=True):
        Frames.__init__(self, filename, gray, invert)
        self.shape = (self.capture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH),
                      self.capture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
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
