import logging
FORMAT = "%(name)s.%(funcName)s:  %(message)s"
logging.basicConfig(level=logging.WARN, format=FORMAT)

from trackpy.api import *
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
