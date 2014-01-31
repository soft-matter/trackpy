import logging
FORMAT = "%(name)s.%(funcName)s:  %(message)s"
logging.basicConfig(level=logging.WARN, format=FORMAT)

from trackpy.api import *
from trackpy.version import version as __version__
