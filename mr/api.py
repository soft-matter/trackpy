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

import numpy as np
from datetime import datetime, timedelta

# SSH-safe matplotlib import
import matplotlib as mpl
import os
if 'DISPLAY' not in os.environ:
    print "No $DISPLAY variable found. Using the Agg matplotlib backend."
    mpl.use('Agg') # suppress plot display

from feature import *
from motion import *
from fitting import *
from plots import *
from rheology import *
from sql import *
from video import *
from tracking import *
from wire import *

import models.power_fluid
