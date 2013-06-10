import os
import cv2
import numpy as np
from PIL import Image
from mr.video.opencv import Video

class PseudoCapture(object):
    def __init__(self, filename):
        self.tiff = Image.open(filename)
        self.count = self._count_frames()
        self.end = False
    def read(self):
        if self.end:
            return False, np.array([])
        image = np.array(self.tiff.getdata()).reshape(self.tiff.size[::-1])
        try:
            self.tiff.seek(self.tiff.tell() + 1)
        except EOFError:
            self.end = True 
        return True, image
    def _count_frames(self):
        count = 0
        while True:
            try:
                self.tiff.seek(count)
                count += 1
            except EOFError:
                break 
        self.tiff.seek(0)
        return count
    def get(self, code):
        if code == cv2.cv.CV_CAP_PROP_FRAME_WIDTH:
            return self.tiff.size[0]
        if code == cv2.cv.CV_CAP_PROP_FRAME_HEIGHT:
            return self.tiff.size[1]
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
