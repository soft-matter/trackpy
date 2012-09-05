from numpy import *
from scipy.interpolate import interp1d
from scipy.stats import nanmean
import pidly
from feature import connect

def fetch(query):
    "Return SQL result set as a numpy array."
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
    """Call Crocker/Weeks track.pro from IDL using pidly module.
    Returns one big array, where the last column is the probe ID."""
    idl = pidly.IDL()
    idl('pt = get_sql("%s")' % query)
    idl('t=track(pt, %s, goodenough=%s, memory=%s)' % 
            tuple(map(str, (max_disp, min_appearances, memory))))
    # 0: x, 1: y, 2: mass, 3: size, 4: ecc, 5: frame, 6: probe_id
    return idl.ev('t')

def slice_by_probe(track_array):
    """Slice the big IDL-style track array into a list of arrays,
    where each array coresponds is a separate probe."""
    probes = []
    probe_count = int(max(track_array[:, 6]))
    # 0: x, 1: y, 2: mass, 3: size, 4: ecc, 5: frame, 6: probe_id
    probes = [track_array[track_array[:, 6] == probe_id][:, 0:6] for \
              probe_id in xrange(probe_count)]
    return probes

def interpolate(original_domain, trajectory, target_domain):
    """Linearly interpolate through gaps in the trajectory
    where the probe was not observed."""
    interpolator = interp1d(probe[:, 5], probe[:, 0:2], bounds_error=False)
    return interpolator(target_domain)

def msd(probe, max_interval):
    max_frame = max(probe[:, 5])
    max_interval = min(max_frame, max_interval)
    domain = arange(0, max_frame)
    interpolated_xy = interpolator(probe[:, 5], probe[:, 0:2], domain)
    msd = []
    for step in xrange(1, max_interval):
        d = diff(interpolated_xy, step, axis=0) \
        sd = d[:,0]**2 + d[1]**2
        msd = nanmean(sd)
        msd_list.append(msd)

def subtract_drift(probes): 
    pass
