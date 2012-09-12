import MySQLdb
import numpy as np

def connect():
    "Return an open connection to the database."
    try:
        conn = MySQLdb.connect(host='localhost', 
                               user='scientist',
                               passwd='scientist', db='exp3')
    except MySQLdb.Error, e:
        print "Cannot connect to database."
        print "Error code:", e.args[0]
        print "Error message:", e.args[1]
    return conn

def fetch(query):
    "Return SQL result set as a numpy array."
    conn = connect()
    c = conn.cursor()
    c.execute(query)
    results = np.array(c.fetchall())
    c.close()
    conn.close()
    autolog("Fetched {} rows".format(c.rowcount))
    return results 

def query_feat(trial, stack, version=None, where=None):
    "Return a query for features from Features."
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
    query = ("SELECT x, y, mass, size, ecc, frame, probe FROM Trajectories "
              "WHERE trial={} AND stack={}".format(trial, stack))
    if where:
        if type(where) is str:
            query += ' AND ' + where
        elif type(where) is list:
            query += ' AND ' + ' AND '.join(where)
    query += " ORDER BY probe, frame"
    return query 

def insert_features(trial, stack, frame, centroids, conn, override=False):
    "Insert centroid information into the MySQL database."
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
            autolog('Overriding')
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
