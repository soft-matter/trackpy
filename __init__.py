import logging
import matplotlib
import os

# Configure logging for all modules in this package.
FORMAT = "%(name)s.%(funcName)s:  %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

# And also use a logger here, for what follows.
logger = logging.getLogger(__name__)

# Choose matplotlib backend before matplotlib.pyplot is imported by the modules.
# If the environmental variable 'ssh' is set, use the ssh-safe backend, Agg. 
# Otherwise, use the backend GTKAgg.
try:
    os.environ['ssh']
    matplotlib.use('Agg')
    logger.info("The matplotlib backend has been set to 'Agg'.")
except KeyError:
    matplotlib.use('GTKAgg')
    logger.info("The matplotlib backend is 'GTKAgg'. If you use "
                "this package over ssh, it is safer to use 'Agg'. Do this by "
                "setting the enviromental variable 'ssh'. "
                "Then restart the interpreter and import this module.")  

__all__ = ["feature", "motion", "sql", "diagnostics", "viscosity"]

