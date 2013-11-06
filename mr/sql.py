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

import os
import numpy as np
import MySQLdb

def connect():
    "Return an open connection to the database."
    try:
        import MySQLdb
    except ImportError:
        logger.error("MySQLdb could not be imported.")
        return None
    try:
        conn = MySQLdb.connect(
            read_default_file=os.path.expanduser('~/.my.cnf'),
            read_default_group='mr')
    except MySQLdb.Error, e:
        logger.error("Cannot connect to database. I look for connection "
                    "parameters in your system's "
                    "mysql default file, usually called ~/.my.cnf. "
                    "Create a group under the heading [mr].")
        logger.error("Error code: %s", e.args[0])
        logger.error("Error message: %s", e.args[1])
        return None
    return conn
