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

def split_by_probe(track_array, keep_traj_only=True):
    """Split the big IDL-style track array into a list of arrays,
    where each array coresponds is a separate probe."""
    boundaries, = where(diff(track_array[:, 6], axis=0) == 1.0)
    boundaries += 1
    if keep_traj_only:
        traj = split(track_array[:, [5, 0, 1]], boundaries)
        # 0: frame, 1: x, 2: y
        return traj
    else: 
        probes = split(track_array[:, :6], boundaries)
        # 0: x, 1: y, 2: mass, 3: size, 4: ecc, 5: frame, 6: probe_id
        return probes

def interpolate(traj):
    """Linearly interpolate through gaps in the trajectory
    where the probe was not observed."""
    # 0: frame, 1: x, 2: y
    first_frame, last_frame = traj[:, 0][[0,-1]]
    full_domain = arange(first_frame, 1 + last_frame)
    interpolator = interp1d(traj[:, 0], traj[:, 1:3], axis=0)
    return column_stack((full_domain, interpolator(full_domain)))

def displacement(a, n):
    """Return difference between nth-order neighbors.
    This is not the same as numpy.diff(a, n), the nth-order derivative."""
    return a[n:]-a[:-n]

def msd(traj, max_interval=None, detail=True):
    """Compute the mean displacement and mean squared displacement of a
    trajectory over a range of time intervals (measured in elapsed frames)."""
    # 0: frame, 1: x, 2: y
    max_interval = max_interval if max_interval else 30 # default
    max_interval = min(max_interval, traj.shape[0])
    intervals = xrange(1, 1 + max_interval)
    traj = interpolate(traj)
    _msd = _detailed_msd if detail else _simple_msd
    results = [_msd(traj, i) for i in intervals]
    return vstack(results)
     
def _detailed_msd(traj, interval):
    """Given a continuous trajectory and a time interval (in frames), 
    return t, <x>, <y>, <r>, <x^2>, <y^2>, <r^2>, N."""
    d = displacement(traj[:, 1:], interval) # [[dx, dy], ...]
    sd = d**2
    stuff = column_stack((d, sum(d, axis=1), sd, sum(sd, axis=1)))
    # [[dx, dy, dr, dx^2, dy^2, dr^2], ...]
    mean_stuff = mean(stuff, axis=0)
    # Estimate statistically independent measurements:
    N = round(2*stuff.shape[0]/float(interval))
    return append(array([interval]), mean_stuff, array([N])) 

def _simple_msd(traj, interval):
    """Given a continuous trajectory and a time interval (in frames),
    return t, <r^2>."""
    d = displacement(traj[:, 1:], interval) # [[dx, dy], ...]
    sd = d**2
    msd = mean(sum(sd, axis=1), axis=0)
    return array([interval, msd]) 

def plot_msds(track_array, max_interval=50):
    msds = [msd(traj, detail=False) for traj in track_array]
    print msds

def subtract_drift(probes): 
    pass
