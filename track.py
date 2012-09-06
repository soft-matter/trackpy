from numpy import *
from scipy.interpolate import interp1d
from scipy.stats import nanmean
import pidly
from feature import connect
from matplotlib.pyplot import *

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
        query = ("SELECT x, y, mass, size, ecc, frame FROM Features WHERE "
                 "trial=%s AND stack=%s AND version=%s " 
                 % tuple(map(str, (trial, stack, version)))) 
    else:
        query = ("SELECT x, y, mass, size, ecc, frame FROM Features WHERE "
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

def split_by_probe(track_array):
    """Split the big IDL-style track array into a list of arrays,
    where each array coresponds is a separate probe."""
    # 0: x, 1: y, 2: mass, 3: size, 4: ecc, 5: frame, 6: probe_id
    indicies, = where(diff(track_array[:, 6], axis=0) == 1.0)
    indicies += 1 # diff offsets indicies to the left by one 
    probes = split(track_array[:, :6], indicies)
    return probes

def interpolate(probe):
    """Linearly interpolate through gaps in the trajectory
    where the probe was not observed."""
    # 0: x, 1: y, 2: mass, 3: size, 4: ecc, 5: frame
    first_frame, last_frame = probe[:, 5][[0,-1]]
    full_domain = arange(first_frame, 1 + last_frame)
    interpolator = interp1d(probe[:, 5], probe[:, 0:2], axis=0)
    return column_stack((full_domain, interpolator(full_domain)))

def dx_dstep(a, step):
    return a[step:]-a[:-step]

# def msd (probe, max_interval):



def msd1(probe, max_interval):
    max_frame = max(probe[:, 5])
    min_frame = min(probe[:, 5])
    max_interval = min(max_frame - min_frame, max_interval)
    domain = arange(min_frame, 1 + max_frame)
    interpolating_func = interp1d(probe[:, 5], probe[:, 0:2], axis=0)
    interpolated_xy = interpolating_func(domain)
    print interpolated_xy.shape
    msd_values = []
    msd_domain = range(1, max_interval)
    for step in msd_domain:
        d = diff(interpolated_xy, n=step, axis=0)
        print d.shape, average(abs(d.flatten()))
        sd = d[:,0]**2 + d[:, 1]**2
        msd = nanmean(sd)
        msd_values.append(msd)
    return msd_domain, msd_values

def plot_msds(probes, max_interval=50):
    for probe in probes:
        msd_domain, msd_values = msd(probe, max_interval)
        plot(log(msd_domain), log(msd_values), '-o')
    show()

def subtract_drift(probes): 
    pass
