import warnings

from mr.video.image_sequence import ImageSequence

def require_cv2_Video(*args, **kwargs):
    raise ImportError("""To import frames from video files, you must install 
OpenCV and the Python module cv2.""")

def require_cv2_tools(*args, **kwargs):
    raise ImportError("""To use video tools, you must install 
OpenCV and the Python module cv2.""")

def require_libtiff(*args, **kwargs):
    raise ImportError("""To import tiff stacks, you must install libtiff.""")

try:
    import cv2
except ImportError:
    Video = require_cv2_Video
    circle = require_cv2_tools
    play = require_cv2_tools
else:
    from mr.video.opencv import Video
    from mr.video.annotate import circle
    from mr.video.playback import play

try:
    import libtiff
except ImportError:
    TiffStack = require_libtiff
else:
    from mr.video.tiff_stack import TiffStack
