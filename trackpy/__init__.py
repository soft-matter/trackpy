from trackpy.api import *
from ._version import get_versions


__version__ = get_versions()['version']
del get_versions


handle_logging()
