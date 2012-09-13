import logging
import matplotlib

# Configure logging for all modules in this package.
FORMAT = "%(name)s.%(funcName)s:  %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

# Some setup.
logger = logging.getLogger(__name__)
matplotlib.use('Agg')
logger.info("The matplotlib backend has been set to 'Agg' because the default "
            "backend causes fatal errors when accessed using ssh.")

__all__ = ["feature", "track", "sql", "diagnostics", "viscosity"]

