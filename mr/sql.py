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

import MySQLdb
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

def connect():
    "Return an open connection to the database."
    try:
        conn = MySQLdb.connect(read_default_group='mr')
    except MySQLdb.Error, e:
        logger.error("Cannot connect to database. I look for connection "
                    "parameters in your system's "
                    "mysql default file, usually called ~/.my.cnf. "
                    "Create a group under the heading [mr].")
        logger.error("Error code: %s", e.args[0])
        logger.error("Error message: %s", e.args[1])
    return conn

def fetch(query):
    "Return SQL result set as a numpy array."
    conn = connect()
    c = conn.cursor()
    c.execute(query)
    results = np.array(c.fetchall())
    c.close()
    conn.close()
    logger.info("Fetched %d rows", c.rowcount)
    return results 

def name_to_id(trial, stack=None):
    """Convert human-readable trial or stack names to their database IDs.
    trial name or ID -> trial ID
    trial name or ID, stack name or ID -> trial ID, stack ID"""
    # Dispatch trivial cases where nothing is done.
    if type(trial) is int and type(stack) is int:
        return trial, stack
    if type(trial) is int and stack is None:
        return trial
    # Now the meat.
    conn = connect ()
    c = conn.cursor()
    if type(trial) is str:
        c.execute("SELECT trial FROM Trials WHERE trial_name=%s", (trial,))
        if c.rowcount == 0:
            raise ValueError, "There is no trial named {}.".format(trial)
        trial, = c.fetchone()
    if stack is None:
        return int(trial)
    elif type(stack) is int:
        return int(trial), stack
    elif type(stack) is str:
        c.execute("SELECT stack FROM Stacks WHERE stack_name=%s", (stack,))
        if c.rowcount == 0:
            raise ValueError, "There is no stack named {}.".format(trial)
        stack, = c.fetchone()
        return int(trial), int(stack)

def new_stack(trial, name, video_file, vstart, duration, start, end):
    "Insert a stack into the database, and return its id number."
    trial = name_to_id(trial)
    conn = connect()
    c = conn.cursor()
    c.execute("""INSERT INTO Stacks (trial, name, video, start, end, """
              """vstart, duration) VALUES """
              """(%s, %s, %s, %s, %s, %s, %s)""",
              (trial, stack_name, video_file, start, end, vstart, duration))
    stack = c.lastrowid
    c.close()
    conn.close()
    logging.info('New stack: trial=%s, stack=%s' % (trial, stack))
    return stack

def query_feat(trial, stack, version=None, where=None):
    "Return a query for features from Features."
    trial, stack = name_to_id(trial, stack)
    if version:
        query = ("SELECT x, y, mass, size, ecc, frame FROM Features WHERE "
                 "trial={} AND stack={} AND version={}".format(
                 trial, stack, version))
    else:
        query = ("SELECT x, y, mass, size, ecc, frame FROM Features WHERE "
                 "trial={} AND stack={}".format(trial, stack))
    if where:
        if type(where) is str:
            query += ' AND ' + where
        elif type(where) is list:
            query += ' AND ' + ' AND '.join(where)
    query += " ORDER BY frame"
    return query 

def query_traj(trial, stack, where=None):
    "Return a query for trajectories from Trajecotires."
    trial, stack = name_to_id(trial, stack)
    query = ("SELECT probe, frame, x, y FROM Trajectories "
              "WHERE trial={} AND stack={}".format(trial, stack))
    if where:
        if type(where) is str:
            query += ' AND ' + where
        elif type(where) is list:
            query += ' AND ' + ' AND '.join(where)
    query += " ORDER BY probe, frame"
    return query 

def insert_feat(trial, stack, frame, centroids, conn, override=False):
    "Insert centroid characteristics into the MySQL database."
    try:
        c = conn.cursor()
        # Load the data in a small temporary table.
        c.execute("CREATE TEMPORARY TABLE NewFeatures"
                  "(x float, y float, mass float, size float, ecc float)")
        c.executemany("INSERT INTO NewFeatures (x, y, mass, size, ecc) "
                      "VALUES (%s, %s, %s, %s, %s)", centroids)
        # In one step, tag all the rows with identifiers (trial, stack, frame).
        # Copy the temporary table into the big table of features.
        c.execute("INSERT INTO Features "
                  "(trial, stack, frame, x, y, mass, size, ecc) "
                  "SELECT %s, %s, %s, x, y, mass, size, ecc FROM NewFeatures" 
                  % (trial, stack, frame))
        c.execute("DROP TEMPORARY TABLE NewFeatures")
        c.close()
    except:
        print sys.exc_info()
        return False
    return True

def feature_duplicate_check(trial, stack, conn):
    "Return false if the database has no entries for this trial and stack."
    c = conn.cursor()
    c.execute("SELECT COUNT(1) FROM Features WHERE trial=%s AND stack=%s",
              (trial, stack))
    count, = c.fetchone()
    return count != 0.0

def insert_traj(trial, stack, track_array, override=False):
    "Insert a track array into the MySQL database."
    if (type(trial) is not int or type(stack) is not int):
        raise TypeError, ("The first two arguments of insert_traj must be the"
                         "trial and stack numbers.")
    conn = connect()
    if traj_duplicate_check(trial, stack, conn):
        if override:
            logger.info("Overriding, and appending on duplicates in database.")
        else:
            raise ValueError, ("There is data in the database for this track"
                              "and stack. Set keyword override=True to proceed.")
            conn.close()
    try:
        c = conn.cursor()
        # Load the data in a small temporary table.
        c.execute("CREATE TEMPORARY TABLE NewTrajectories"
                  "(probe int unsigned, frame int unsigned, "
                  "x float, y float, mass float, size float, ecc float)")
        c.executemany("INSERT INTO NewTrajectories "
                      "(x, y, mass, size, ecc, frame, probe) "
                      "VALUES (%s, %s, %s, %s, %s, %s, %s)", 
                      map(tuple, list(track_array)))
        # In one step, tag all the rows with identifiers (trial, stack, frame).
        # Copy the temporary table into the big table of features.
        c.execute("INSERT INTO Trajectories "
                  "(trial, stack, probe, frame, x, y, mass, size, ecc) "
                  "SELECT %s, %s, probe, frame, x, y, mass, size, ecc "
                  "FROM NewTrajectories", (trial, stack))
        c.execute("DROP TEMPORARY TABLE NewTrajectories")
        c.close()
    except MySQLdb.Error, e:
        print e

def traj_duplicate_check(trial, stack, conn):
    "Return false if the database has no entries for this trial and stack."
    c = conn.cursor()
    c.execute("SELECT COUNT(1) FROM Trajectories WHERE trial=%s AND stack=%s",
              (trial, stack))
    count, = c.fetchone()
    return count != 0.0
