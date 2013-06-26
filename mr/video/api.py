import warnings

from mr.video.image_sequence import ImageSequence

try:
    import cv2
except ImportError:
    warnings.warn(
        """The module cv2 could not be found. All dependent video tools
        will not be imported.""")
else:
    from mr.video.opencv import Video
    from mr.video.trace import circle
    from mr.video.basic import play

try:
    import libtiff
except ImportError:
    warnings.warn(
        """The module libtiff could not be found. If you wish you load
        multi-frame tiff files using mr.TiffStack, you must first install
        libtiff.""")
else:
    from mr.video.tiff import TiffStack

# Legacy
# from muxing import *
