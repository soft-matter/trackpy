from muxing import *

try:
    import cv2
    from opencv import *
except ImportError:
    raise UserWarning, \
    """The module cv2 could not be found. All dependent video tools
    will not be imported."""

try:
    from PIL import Image
    from tif import *
except ImportError:
    raise UserWarning, \
    """The module PIL could not be found. All dependent video tools
    will not be imported."""
  
