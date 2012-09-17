import logging
import matplotlib
import os
import warnings

# Configure logging for all modules in this package.
FORMAT = "%(name)s.%(funcName)s:  %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

# Choose matplotlib backend before matplotlib.pyplot is imported by the modules.
# If the environmental variable 'ssh' is set, use the ssh-safe backend, Agg. 
# Otherwise, use the backend GTKAgg.
with warnings.catch_warnings(): # Catch warning on reload(mr).
    try:
        os.environ['ssh']
        matplotlib.use('Agg')
    except KeyError:
        matplotlib.use('GTKAgg')

from mr.api import *
