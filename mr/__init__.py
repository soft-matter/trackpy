# Copyright 2012 Daniel B. Allan
# dallan@pha.jhu.edu, daniel.b.allan@gmail.com
# http://pha.jhu.edu/~dallan
# http://www.danallan.com
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses>.

import logging
import os
import warnings

# Configure logging for all modules in this package.
FORMAT = "%(name)s.%(funcName)s:  %(message)s"
logging.basicConfig(level=logging.WARN, format=FORMAT)

import numpy as np
import pandas as pd

# SSH-safe matplotlib import
import matplotlib as mpl
import os
if 'DISPLAY' not in os.environ:
    print "No $DISPLAY variable found. Using the Agg matplotlib backend."
    mpl.use('Agg') # suppress plot display

from mr.api import *
from mr.video.api import *
from mr import wire

from mr.version import version as __version__
