import logging
FORMAT = "%(name)s.%(funcName)s:  %(message)s"
logging.basicConfig(level=logging.WARN, format=FORMAT)

from .api import *
from .version import version as __version__
