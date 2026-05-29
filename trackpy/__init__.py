# Configure a logger from trackpy.
# This must be done before utils is imported.
import logging
logger = logging.getLogger(__name__)


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from trackpy.api import *

handle_logging()
