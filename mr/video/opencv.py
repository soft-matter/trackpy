import os
import cv2
import numpy as np

def open_video(filename):
    """Thin convenience function for return an opencv2 Capture object."""
    # ffmpeg -i unreadable.avi -sameq -r 30 readable.avi
    if not os.path.isfile(filename):
        raise ValueError, "%s is not a file." % filename
    capture = cv2.VideoCapture(filename)
    return capture

class Video(object):
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
    >>> imshow(video.next()) # Show the first frame.
    >>> imshow(video.next()[0:10][0:10]) # Show one corner of the second frame.
    >>> video.rewind()
    >>> imshow(video.next()) # First frame again.

    >>> for frame in video:
    ...    # Do something with every frame.
 
    >>> for frame in video[10:20]:
    ...    # Do something with frames 10-20.
 
    >>> frame_count = video.count # Number of frames in video
    """
    
    def __init__(self, filename, gray=True, invert=True):
        self.filename = filename
        self.gray = gray
        self.invert = invert
        self.capture = open_video(self.filename)
        self.cursor = 0
        self.count = self._count()
        self.endpoint = None

    def __repr__(self):
        return """Video Frames
Source File: %s
Cursor at Frame %d of %d""" % (self.filename, self.cursor, self.count)

    def __iter__(self):
        return self

    def _process(self, frame):
        if self.gray:
            frame = cv2.cvtColor(frame, cv2.cv.CV_RGB2GRAY)
        if self.invert:
            frame *= -1
            frame += 255
        return frame 

    @property
    def endpoint(self):
        return self._endpoint

    @endpoint.setter
    def endpoint(self, val):
        self._endpoint = val

    def _count(self):
        "Return total frame count. Result is not always exact."
        return int(self.capture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    def seek_forward(self, val):
        for _ in range(val):
            self.next()
        
    def rewind(self):
        """Reopen the video file to start at the beginning. ('Seeking'
        capabilities in the underlying OpenCV library are not reliable.)"""
        self.capture = open_video(self.filename)
        self.cursor = 0

    def next(self):
        if self.endpoint is not None and self.cursor > self.endpoint:
            raise StopIteration
        return_code, frame = self.capture.read()
        if not return_code:
            # A failsafe: the frame count is not always accurate.
            raise StopIteration
        frame = self._process(frame)
        self.cursor += 1
        return frame

    def __getitem__(self, val):
        if isinstance(val, slice):
            start, stop, step = val.indices(self.count)
            if step != 1:
                raise NotImplementedError, \
                    "Step must be 1."
        else:
            start = val
            stop = None    
        video_copy = Video(self.filename, self.gray, self.invert)
        video_copy.seek_forward(start)
        video_copy.endpoint = stop
        return video_copy

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
