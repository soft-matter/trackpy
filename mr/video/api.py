try:
    import cv2
    from mr.video.opencv import Video
    from mr.video.trace import circle
except ImportError:
    raise UserWarning, \
    """The module cv2 could not be found. All dependent video tools
    will not be imported."""

from tif import tif_frames

# Legacy
# from muxing import *
