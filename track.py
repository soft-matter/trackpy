from numpy import *
from scipy.interpolate import interp1d
from scipy.stats import nanmean
import pidly
from connect import sql_connect
from matplotlib.pyplot import *
from scipy.stats import linregress

def autolog(message):
    "Write a message to the log, with the calling function's name."
    import inspect, logging
    func = inspect.currentframe().f_back.f_code
    logging.info("%s: %s" % (
        func.co_name, 
        message
    ))

def sql_fetch(query):
    "Return SQL result set as a numpy array."
    conn = sql_connect()
    c = conn.cursor()
    c.execute(query)
    results = array(c.fetchall())
    c.close()
    conn.close()
    autolog("Fetched {} rows".format(c.rowcount))
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
    t = idl.ev('t')
    idl.close()
    return t

def sql_insert(trial, stack, track_array, override=False):
    "Insert a track array into the MySQL database."
    conn = sql_connect()
    if sql_duplicate_check(trial, stack, conn):
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
    boundaries, = where(diff(track_array[:, 6], axis=0) > 0.0)
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

def msd(traj, microns_per_px=100/427., fps=30., 
        max_interval=None, detail=False):
    """Compute the mean displacement and mean squared displacement of a
    trajectory over a range of time intervals. Input in units of px and frames;
    output in units of microns and seconds."""
    # 0: frame, 1: x, 2: y
    max_interval = max_interval if max_interval else 50 # default
    max_interval = min(max_interval, traj.shape[0])
    intervals = xrange(1, 1 + max_interval)
    traj = interpolate(traj)
    _msd = _detailed_msd if detail else _simple_msd
    results = [_msd(traj, i, microns_per_px, fps) for i in intervals]
    return vstack(results)
     
def _detailed_msd(traj, interval, microns_per_px, fps):
    """Given a continuous trajectory and a time interval (in frames), 
    return t, <x>, <y>, <r>, <x^2>, <y^2>, <r^2>, N."""
    d = displacement(microns_per_px*traj[:, 1:], interval) # [[dx, dy], ...]
    sd = d**2
    stuff = column_stack((d, sum(d, axis=1), sd, sum(sd, axis=1)))
    # [[dx, dy, dr, dx^2, dy^2, dr^2], ...]
    mean_stuff = mean(stuff, axis=0)
    # Estimate statistically independent measurements:
    N = round(2*stuff.shape[0]/float(interval))
    return append(array([interval])/float(fps), mean_stuff, array([N])) 

def _simple_msd(traj, interval, microns_per_px, fps):
    """Given a continuous trajectory and a time interval (in frames),
    return t, <r^2>."""
    d = displacement(microns_per_px*traj[:, 1:], interval) # [[dx, dy], ...]
    sd = d**2
    msd_result = mean(sum(sd, axis=1), axis=0)
    return array([interval/float(fps), msd_result]) 

def ensemble_msd(flexible_input, microns_per_px=100/427., fps=30.):
    """Return ensemble mean squared displacement. Input in units of px
    and frames. Output in units of microns and seconds."""
    probes = _validate_input(flexible_input)
    m = vstack([msd(traj, microns_per_px, fps, detail=False) \
                for traj in probes])
    m = m[m[:, 0].argsort()] # sort by dt 
    boundaries, = where(diff(m[:, 0], axis=0) > 0.0)
    boundaries += 1
    m = split(m, boundaries) # list of arrays, one for each dt
    ensm_m = vstack([mean(this_m, axis=0) for this_m in m])
    power, coeff = fit_powerlaw(ensm_m)
    print 'Power Law n =', power
    print 'D =', coeff/4.
    return ensm_m

def fit_powerlaw(a):
    "Fit a power law to MSD data. Return the power and the coefficient."
    # This is not a generic power law. By treating it as a linear regression in
    # log space, we assume no additive constant: y = 0 + coeff*x**power.
    slope, intercept, r, p, stderr =  linregress(log(a[:, 0]), log(a[:, 1]))
    return slope, exp(intercept)

def drift(flexible_input, suppress_plot=False):
    "Return the ensemble drift, x(t)."
    x_list = _validate_input(flexible_input, output_style='probes')
    dx_list = [column_stack(
               (diff(x[:, 0]), x[1:, 0], diff(x[:, 1:], axis=0))
               ) for x in x_list] # dt, t, dx, dy
    dx = vstack(dx_list) # dt, t, dx, dy
    dx = dx[dx[:, 0] == 1.0, 1:] # Drop entries where dt > 1 ( gap).
    dx = dx[dx[:, 0].argsort()] # sort by t
    boundaries, = where(diff(dx[:, 0], axis=0) > 0.0)
    boundaries += 1
    dx_list = split(dx, boundaries) # list of arrays, one for each t
    ensemble_dx = vstack([mean(dx, axis=0) for dx in dx_list])
    ensemble_dx = interpolate(ensemble_dx) # Fill in any gaps.
    # ensemble_dx is t, dx, dy. Integrate to get t, x, y.
    x = column_stack((ensemble_dx[:, 0], cumsum(ensemble_dx[:, 1:], axis=0)))
    if not suppress_plot:
        plot(x[:, 0], x[:, 1], '-', label='X')
        plot(x[:, 0], x[:, 2], '-', label='Y')
        xlabel('time [frames]')
        ylabel('drift [px]')
        legend(loc='best')
        show()
    return x 

def subtract_drift(flexible_input, d=None):
    "Return a copy of the track_array with the overall drift subtracted out."
    track_array = _validate_input(flexible_input, 'track array')
    if d is None: 
        d=drift(track_array, suppress_plot=True)
    new_ta = copy(track_array)
    for t, x, y in d:
        new_ta[new_ta[:, 5] == t, 0:2] -= [x, y] 
    # 0: x, 1: y, 2: mass, 3: size, 4: ecc, 5: frame, 6: probe_id
    return new_ta

def is_localized(probe, threshold=0.4):
    "Is this probe's motion localized?"
    m = msd(probe)
    power, coeff = fit_powerlaw(m)
    if power < threshold: return True
    return False

def is_diffusive(probe, threshold=0.85):
    "Is this probe's motion diffusive?"
    m = msd(probe)
    power, coeff = fit_powerlaw(m)
    if power > threshold: return True
    return False

def is_unphysical(probe, threshold=0.08):
    """Is the first MSD datapoint unphysically high? (This is sometimes an
    artifact of uneven drift.)"""
    m = msd(probe)
    if m[0, 1] >  threshold: return True
    return False

def split_branches(probes, threshold=0.85, lower_threshold=0.4):
    "Sort list of probes into three lists, sorted by mobility."
    probes = _validate_input(probes)
    diffusive = [p for p in probes if is_diffusive(p)]
    localized = [p for p in probes if is_localized(p)]
    subdiffusive = [p for p in probes if ((not is_localized(p)) and \
                           (not is_diffusive(p)))]
    autolog("{} diffusive, {} localized, {} subdiffusive".format(
                 len(diffusive), len(localized), len(subdiffusive)))
    return diffusive, localized, subdiffusive

def plot_traj(probes, superimpose=None, microns_per_px=100/427.):
    """Plot traces of trajectories for each probe.
    Optionally superimpose it on a fram from the video."""
    probes = _validate_input(probes)
    if superimpose:
        image = 1-imread(superimpose)
        imshow(image, cmap=cm.gray)
        xlim(0, image.shape[1])
        ylim(0, image.shape[0])
        autolog("Using units of px, not microns.")
        microns_per_px = 1
        xlabel('x [px]')
        ylabel('y [px]')
    else:
        xlabel('x [um]')
        ylabel('y [um]')
    for traj in probes:
        plot(microns_per_px*traj[:, 1], microns_per_px*traj[:, 2])
    show()

def _validate_input(flexible_input, output_style='probes'):
    """Accept either the IDL-style track_array or a list of probes,
    and return one or the other."""
    if output_style == 'track array':
        if type(flexible_input) is ndarray:
            return flexible_input
        elif type(flexible_input) is list:
            return vstack(probes)
        else:
            raise TypeError, ("Input must be either the ndarray track_array"
                              "or the list of probes.")
    elif output_style == 'probes':
        if type(flexible_input) is list:
            return flexible_input
        elif type(flexible_input) is ndarray:
            return split_by_probe(flexible_input)
        else:
            raise TypeError, ("Input must be either the ndarray track_array "
                              "or the list of probes.")
    else:
        raise ValueError, "output_style must be 'track array' or 'probes'."

def plot_msd(probes, max_interval=None,
             microns_per_px=100/427., fps=30., 
             indv=True, ensm=False, branch=False, powerlaw=True,
             defer=False, suppress_labels=False):
    "Plot individual MSDs for each probe, or ensemble MSD, or both."
    probes = _validate_input(probes)
    if (indv and not branch):
        msds = [msd(traj, microns_per_px, fps, max_interval, detail=False) \
                for traj in probes] 
        for counter, m in enumerate(msds):
            # Label only one instance for the plot legend.
            if counter == 0:
                if not suppress_labels:
                    loglog(m[:, 0], m[:, 1], 'k.-', alpha=0.3,
                           label='individual probe MSDs')
            else:
                loglog(m[:, 0], m[:, 1], 'k.-', alpha=0.3)
    if ensm:
        m = ensemble_msd(probes)
        if not suppress_labels:
            loglog(m[:, 0], m[:, 1], 'ro-', linewidth=3, label='ensemble MSD')
        else:
            loglog(m[:, 0], m[:, 1], 'ro-', linewidth=3)
        if powerlaw:
            power, coeff = fit_powerlaw(m)
            loglog(m[:, 0], coeff*m[:, 0]**power, '-', color='#019AD2', linewidth=2,
                   label=_powerlaw_label(power, coeff))
    if branch:
        upper_branch, lower_branch, middle_branch = split_branches(probes)
        plot_msd(upper_branch, indv=True, ensm=True, powerlaw=True, 
                 defer=True)
        plot_msd(middle_branch, indv=True, ensm=False, powerlaw=False, 
                 defer=True, suppress_labels=True)
        plot_msd(lower_branch, indv=True, ensm=True, powerlaw=True,
                 suppress_labels=True)
        return
    # Label ticks with plain numbers, not scientific notation:
    gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ylim(0.01, 100)
    print 'Limits of y range are manually set to {} - {}.'.format(*ylim())
    xlabel('lag time [s]')
    ylabel('msd [um$^2$]')
    if not defer:
        legend(loc='upper left')
        show()

def _powerlaw_label(power, coeff):
    """Return a string suitable for a legend label, including power
    and D if motion is diffusive, but only power if it is subdiffusive."""
    DIFFUSIVE_THRESHOLD = 0.90
    label = 'power law fit\nn=' + '{:.2f}'.format(power)
    if power >= DIFFUSIVE_THRESHOLD:
        label += '  D=' + '{:.3f}'.format(coeff/4) + ' um$^2$/s'
    return label
    
    
