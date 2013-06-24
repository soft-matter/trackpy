import os
import numpy as np
import collections

class Frames(object):
    
    def __init__(self, filename, gray=True, invert=True):
        self.filename = filename
        self.gray = gray
        self.invert = invert
        self.capture = self._open(self.filename)
        self.cursor = 0
        self.endpoint = None
        # Subclass will specify self.count and self.shape.

    def __repr__(self):
        return """<Frames>
Source File: %s
Frame Dimensions: %d x %d
Cursor at Frame %d of %d""" % (self.filename, self.shape[0], self.shape[1],
                               self.cursor, self.count)

    def __iter__(self):
        return self

    @property
    def endpoint(self):
        return self._endpoint

    @endpoint.setter
    def endpoint(self, val):
        self._endpoint = val

    def seek_forward(self, val):
        for _ in range(val):
            self.next()
        
    def rewind(self):
        """Reopen the video file to start at the beginning. ('Seeking'
        capabilities in the underlying OpenCV library are not reliable.)"""
        self.capture = self._open(self.filename)
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
        if isinstance(val, int):
            if val > self.cursor:
                self.seek_forward(val - self.cursor)
                return self.next()
            elif self.cursor == val:
                return self.next()
            else:
                video_copy = self.__class__(self.filename, 
                                            self.gray, self.invert)
                video_copy.seek_forward(val)
                return video_copy.next()
        if isinstance(val, slice):
            start, stop, step = val.indices(self.count)
            if step != 1:
                raise NotImplementedError, \
                    "Step must be 1."
        elif isinstance(val, collections.Iterable):
            return (self[i] for i in val)
            start = val
            stop = None    
        video_copy = self.__class__(self.filename, self.gray, self.invert)
        video_copy.seek_forward(start)
        video_copy.endpoint = stop
        return video_copy
