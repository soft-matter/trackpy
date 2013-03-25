import cv
from PIL import Image
import numpy as np

def invert_in_place(a):
    a *= -1
    a += 255

def open_video(filename, gray=True, invert=True):
    # ffmpeg -i unreadable.avi -sameq -r 30 readable.avi
    capture = cv.CaptureFromFile(filename)
    def frame_generator():
        frame = cv.QueryFrame(capture)
        a = np.asarray(cv.GetMat(frame))
        if invert:
            invert_in_place(a)
        return a
    def gray_frame_generator():
        frame = cv.QueryFrame(capture)
        size = frame.width, frame.height
        gray_frame = cv.CreateImage(size, frame.depth, 1)
        cv.CvtColor(frame, gray_frame, cv.CV_RGB2GRAY)
        a = np.asarray(cv.GetMat(gray_frame))
        if invert:
            invert_in_place(a)
        return a
    return gray_frame_generator if gray else frame_generator
