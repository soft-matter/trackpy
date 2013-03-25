import cv
import numpy as np

def open_video(filename):
    # ffmpeg -i unreadable.avi -sameq -r 30 readable.avi
    capture = cv.CaptureFromFile(filename)
    def frame_generator():
        img = np.asarray(cv.GetMat(cv.QueryFrame(capture)))
        return img 
    return frame_generator
