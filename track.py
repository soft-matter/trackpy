from numpy import *
from scipy.interpolate import interp1d
from scipy.stats import nanmean
import pidly
from connect import sql_connect
from matplotlib.pyplot import *

def sql_fetch(query):
    "Return SQL result set as a numpy array."
    conn = sql_connect()
    c = conn.cursor()
    c.execute(query)
    results = array(c.fetchall())
    c.close()
    conn.close()
    print c.rowcount
    return results 

def query_feat(trial, stack, version=None, where=None):
    "Return a query for features from Features."
    if version:
        query = ("SELECT x, y, mass, size, ecc, frame FROM Features WHERE "
                 "trial=%s AND stack=%s AND version=%s"
                 % tuple(map(str, (trial, stack, version))))
    else:
        query = ("SELECT x, y, mass, size, ecc, frame FROM Features WHERE "
                 "trial=%s AND stack=%s"
                 % tuple(map(str, (trial, stack))))
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
              "WHERE trial=%s AND stack=%s" %
              tuple(map(str, (trial, stack))))
    if where:
        if type(where) is str:
            query += ' AND ' + where
        elif type(where) is list:
            query += ' AND ' + ' AND '.join(where)
    query += " ORDER BY probe, frame"
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

def sql_insert(trial, stack, track_array, override=False):
    "Insert a track array into the MySQL database."
    conn = sql_connect()
    if sql_duplicate_check(trial, stack, conn):
        if override:
            print 'Overriding'
        else:
            print 'There are entries for this trial and stack already.'
            conn.close()
            return False
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
    except:
        print sys.exc_info()
        return False
    return True

def sql_duplicate_check(trial, stack, conn):
    "Return false if the database has no entries for this trial and stack."
    c = conn.cursor()
    c.execute("SELECT COUNT(1) FROM Trajectories WHERE trial=%s AND stack=%s",
              (trial, stack))
    count, = c.fetchone()
    return count != 0.0

def split_by_probe(track_array, traj_only=True):
    """Split the big IDL-style track array into a list of arrays,
    where each array coresponds is a separate probe."""
    boundaries, = where(diff(track_array[:, 6], axis=0) == 1.0)
    boundaries += 1
    if traj_only:
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

def displacement(x, dt):
    """Return difference between neighbors separated by dt steps (frames).
    This is not the same as numpy.diff(x, n), the nth-order derivative."""
    return x[dt:]-x[:-dt]

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
    msd_result = mean(sum(sd, axis=1), axis=0)
    return array([interval, msd_result]) 

def ensemble_msd(track_array):
    "Return ensemble mean squared displacement."
    m = vstack([msd(traj, detail=False) \
                   for traj in split_by_probe(track_array)])
    m = m[m[:, 0].argsort()] # sort by dt 
    boundaries, = where(diff(m[:, 0], axis=0) > 0.0)
    boundaries += 1
    m = split(m, boundaries) # list of arrays, one for each dt
    ensm_m = vstack([mean(this_m, axis=0) for this_m in m])
    return ensm_m

def drift(track_array, suppress_plot=False):
    "Return the ensemble drift, dx(t)."
    x_list = [interpolate(traj) for traj in split_by_probe(track_array)]
    # t, x, y
    dx_list = [column_stack((x[:, 0], x[:, 1:] - x[0, 1:])) for x in x_list]
    # t, v_x, v_y
    dx = vstack(dx_list)
    dx = dx[dx[:, 0].argsort()] # sort by t
    boundaries, = where(diff(dx[:, 0], axis=0) > 0.0)
    boundaries += 1
    dx_list = split(dx, boundaries) # list of arrays, one for each t
    ensm_dx = vstack([mean(dx, axis=0) for dx in dx_list])
    if not suppress_plot:
        plot(ensm_dx[:, 0], ensm_dx[:, 1], '-', label='X')
        plot(ensm_dx[:, 0], ensm_dx[:, 2], '-', label='Y')
        xlabel('time [frames]')
        ylabel('drift [px]')
        legend(loc='best')
        show()
    return ensm_dx

def subtract_drift(track_array, d=None):
    "Return a copy of the track_array with the overall drift subtracted out."
    if d is None: 
        d=drift(track_array, suppress_plot=True)
    new_ta = copy(track_array)
    for t, x, y in d:
        new_ta[new_ta[:, 5] == t, 0:2] -= [x, y] 
    # 0: x, 1: y, 2: mass, 3: size, 4: ecc, 5: frame, 6: probe_id
    return new_ta

def plot_traj(track_array, microns_per_px=100/427.):
    "Plot traces of trajectories for each probe."
    for traj in split_by_probe(track_array):
        plot(microns_per_px*traj[:, 1], microns_per_px*traj[:, 2])
    xlabel('x [um]')
    ylabel('y [um]')
    show()

def plot_msd(track_array, max_interval=None,
              microns_per_px=100/427., fps=30., ens=False):
    "Plot individual MSDs for each probe. Optionally overlay ensemble MSD."
    msds = [msd(traj, max_interval, detail=False) \
            for traj in split_by_probe(track_array)]
    for m in msds:
        loglog(microns_per_px*m[:, 0], m[:, 1]/float(fps), 'k.-')
    if ens:
        m = ensemble_msd(track_array)
        loglog(microns_per_px*m[:, 0], m[:, 1]/float(fps), 'ro-')
    gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    xlabel('lag time [s]')
    ylabel('msd [um]')
    show()
