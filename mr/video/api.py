try:
    import cv2
except ImportError:
    raise UserWarning, \
    """The module cv2 could not be found. All dependent video tools
    will not be imported."""
else:
    from mr.video.opencv import Video, Frames
    from mr.video.trace import circle
    from mr.video.basic import play

# from tiff import TiffStack

# Legacy
# from muxing import *
