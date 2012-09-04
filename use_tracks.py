from numpy import *
import pidly
from feature import connect

def fetch(query):
    "Return SQL results set as a numpy array."
    conn = connect()
    c = conn.cursor()
    c.execute(query)
    features = array(c.fetchall())
    c.close()
    conn.close()
    print c.rowcount
    return features

def query(trial, stack, version=None, where=None):
    "Return a query for features from UFeature."
    if version:
        query = ("SELECT x, y, mass, size, ecc, frame FROM UFeature WHERE "
                 "trial=%s AND stack=%s AND version=%s " 
                 % tuple(map(str, (trial, stack, version)))) 
    else:
        query = ("SELECT x, y, mass, size, ecc, frame FROM UFeature WHERE "
                 "trial=%s AND stack=%s " 
                 % tuple(map(str, (trial, stack))))
    if where:
        if type(where) is str:
            query += ' AND ' + where
        elif type(where) is list:
            query += ' AND ' + ' AND '.join(where)
    query += " ORDER BY frame"
    return query 

def track(query, max_disp, min_appearances, memory=3):
     "Call Crocker/Weeks track.pro from IDL using pidly module."
     idl = pidly.IDL()
     idl('pt = get_sql("%s")' % query)
     idl('t=track(pt, %s, goodenough=%s, memory=%s)' % 
             tuple(map(str, (max_disp, min_appearances, memory))))
     return idl.ev('t')

def interpolate(tracks):
    "Fill in gaps in a time series."
    pass

def getdx(tracks, step):
    "I think this is just n-order differencing."
    pass
