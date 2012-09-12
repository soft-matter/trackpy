import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import interpolate
import pidly
import diagnostics

def autolog(message):
    "Write a message to the log, with the calling function's name."
    import inspect, logging
    func = inspect.currentframe().f_back.f_code
    logging.info("%s: %s" % (
        func.co_name, 
        message
    ))

def idl_track(query, max_disp, min_appearances, memory=3):
    """Call Crocker/Weeks track.pro from IDL using pidly module.
    Returns one big array, where the last column is the probe ID."""
    idl = pidly.IDL()
    idl('pt = get_sql("{}")'.format(query))
    idl('t=track(pt, {}, goodenough={}, memory={})'.format(max_disp, min_appearances, memory))
    # 0: x, 1: y, 2: mass, 3: size, 4: ecc, 5: frame, 6: probe_id
    t = idl.ev('t')
    idl.close()
    return t

def split_by_probe(track_array, traj_only=True):
    """Split the big IDL-style track array into a list of arrays,
    where each array coresponds is a separate probe."""
    boundaries, = np.where(np.diff(track_array[:, 6], axis=0) > 0.0)
    boundaries += 1
    if traj_only:
        traj = np.split(track_array[:, [5, 0, 1]], boundaries)
        # 0: frame, 1: x, 2: y
        return traj
    else: 
        probes = np.split(track_array[:, :6], boundaries)
        # 0: x, 1: y, 2: mass, 3: size, 4: ecc, 5: frame, 6: probe_id
        return probes

def interp(traj):
    """Linearly interpolate through gaps in the trajectory
    where the probe was not observed."""
    # 0: frame, 1: x, 2: y
    first_frame, last_frame = traj[:, 0][[0,-1]]
    full_domain = np.arange(first_frame, 1 + last_frame)
    interpolator = interpolate.interp1d(traj[:, 0], traj[:, 1:3], axis=0)
    return np.column_stack((full_domain, interpolator(full_domain)))

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
    traj = interp(traj)
    _msd = _detailed_msd if detail else _simple_msd
    results = [_msd(traj, i, microns_per_px, fps) for i in intervals]
    return np.vstack(results)
     
def _detailed_msd(traj, interval, microns_per_px, fps):
    """Given a continuous trajectory and a time interval (in frames), 
    return t, <x>, <y>, <r>, <x^2>, <y^2>, <r^2>, N."""
    d = displacement(microns_per_px*traj[:, 1:], interval) # [[dx, dy], ...]
    sd = d**2
    stuff = np.column_stack((d, np.sum(d, axis=1), sd, np.sum(sd, axis=1)))
    # [[dx, dy, dr, dx^2, dy^2, dr^2], ...]
    mean_stuff = np.mean(stuff, axis=0)
    # Estimate statistically independent measurements:
    N = np.round(2*stuff.shape[0]/float(interval))
    return np.append(np.array([interval])/float(fps), mean_stuff, np.array([N])) 

def _simple_msd(traj, interval, microns_per_px, fps):
    """Given a continuous trajectory and a time interval (in frames),
    return t, <r^2>."""
    d = displacement(microns_per_px*traj[:, 1:], interval) # [[dx, dy], ...]
    sd = d**2
    msd_result = np.mean(np.sum(sd, axis=1), axis=0)
    return np.array([interval/float(fps), msd_result]) 

def ensemble_msd(flexible_input, microns_per_px=100/427., fps=30.):
    """Return ensemble mean squared displacement. Input in units of px
    and frames. Output in units of microns and seconds."""
    probes = _validate_input(flexible_input)
    m = np.vstack([msd(traj, microns_per_px, fps, detail=False) \
                for traj in probes])
    m = m[m[:, 0].argsort()] # sort by dt 
    boundaries, = np.where(np.diff(m[:, 0], axis=0) > 0.0)
    boundaries += 1
    m = np.split(m, boundaries) # list of arrays, one for each dt
    ensm_m = np.vstack([np.mean(this_m, axis=0) for this_m in m])
    power, coeff = fit_powerlaw(ensm_m)
    print 'Power Law n =', power
    print 'D =', coeff/4.
    return ensm_m

def fit_powerlaw(a):
    "Fit a power law to MSD data. Return the power and the coefficient."
    # This is not a generic power law. By treating it as a linear regression in
    # log space, we assume no additive constant: y = 0 + coeff*x**power.
    slope,intercept,r,p,stderr = stats.linregress(log(a[:, 0]), log(a[:, 1]))
    return slope, exp(intercept)

def drift(flexible_input, suppress_plot=False):
    "Return the ensemble drift, x(t)."
    x_list = _validate_input(flexible_input, output_style='probes')
    dx_list = [np.column_stack(
               (np.diff(x[:, 0]), x[1:, 0], np.diff(x[:, 1:], axis=0))
               ) for x in x_list] # dt, t, dx, dy
    dx = np.vstack(dx_list) # dt, t, dx, dy
    dx = dx[dx[:, 0] == 1.0, 1:] # Drop entries where dt > 1 ( gap).
    dx = dx[dx[:, 0].argsort()] # sort by t
    boundaries, = np.where(np.diff(dx[:, 0], axis=0) > 0.0)
    boundaries += 1
    dx_list = np.split(dx, boundaries) # list of arrays, one for each t
    ensemble_dx = np.vstack([np.mean(dx, axis=0) for dx in dx_list])
    ensemble_dx = interp(ensemble_dx) # Fill in any gaps.
    # ensemble_dx is t, dx, dy. Integrate to get t, x, y.
    x = np.column_stack((ensemble_dx[:, 0], 
                         np.cumsum(ensemble_dx[:, 1:], axis=0)))
    if not suppress_plot: plot_drift(x)
    return x 

def plot_drift(x, finish=True, label=''):
    """Plot ensemble drift. To compare drifts of subsets, call multiple times
    with finish=False for all but the last call."""
    plt.plot(x[:, 0], x[:, 1], '-', label=label + ' X')
    plt.plot(x[:, 0], x[:, 2], '-', label=label + ' Y')
    if finish:
        plt.xlabel('time [frames]')
        plt.ylabel('drift [px]')
        plt.legend(loc='best')
        plt.show()

def subtract_drift(flexible_input, d=None):
    "Return a copy of the track_array with the overall drift subtracted out."
    track_array = _validate_input(flexible_input, 'track array')
    if d is None: 
        d=drift(track_array, suppress_plot=True)
    new_ta = np.copy(track_array)
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
        image = 1-plt.imread(superimpose)
        plt.imshow(image, cmap=cm.gray)
        plt.xlim(0, image.shape[1])
        plt.ylim(0, image.shape[0])
        autolog("Using units of px, not microns.")
        microns_per_px = 1
        plt.xlabel('x [px]')
        plt.ylabel('y [px]')
    else:
        plt.xlabel('x [um]')
        plt.ylabel('y [um]')
    for traj in probes:
        plt.plot(microns_per_px*traj[:, 1], microns_per_px*traj[:, 2])
    show()

def _validate_input(flexible_input, output_style='probes'):
    """Accept either the IDL-style track_array or a list of probes,
    and return one or the other."""
    if output_style == 'track array':
        if type(flexible_input) is np.ndarray:
            return flexible_input
        elif type(flexible_input) is list:
            return np.vstack(flexible_input)
        else:
            raise TypeError, ("Input must be either the np.ndarray track_array "
                              "or the list of probes.")
    elif output_style == 'probes':
        if type(flexible_input) is list:
            return flexible_input
        elif type(flexible_input) is np.ndarray:
            return split_by_probe(flexible_input)
        else:
            raise TypeError, ("Input must be either the np.ndarray track_array "
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
                    plt.loglog(m[:, 0], m[:, 1], 'k.-', alpha=0.3,
                           label='individual probe MSDs')
            else:
                plt.loglog(m[:, 0], m[:, 1], 'k.-', alpha=0.3)
    if ensm:
        m = ensemble_msd(probes)
        if not suppress_labels:
            plt.loglog(m[:, 0], m[:, 1], 'ro-', linewidth=3, label='ensemble MSD')
        else:
            plt.loglog(m[:, 0], m[:, 1], 'ro-', linewidth=3)
        if powerlaw:
            power, coeff = fit_powerlaw(m)
            plt.loglog(m[:, 0], coeff*m[:, 0]**power, '-', color='#019AD2', linewidth=2,
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
    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    plt.ylim(0.01, 100)
    print 'Limits of y range are manually set to {} - {}.'.format(*ylim())
    plt.xlabel('lag time [s]')
    plt.ylabel('msd [um$^2$]')
    if not defer:
        plt.legend(loc='upper left')
        plt.show()

def _powerlaw_label(power, coeff):
    """Return a string suitable for a legend label, including power
    and D if motion is diffusive, but only power if it is subdiffusive."""
    DIFFUSIVE_THRESHOLD = 0.90
    label = 'power law fit\nn=' + '{:.2f}'.format(power)
    if power >= DIFFUSIVE_THRESHOLD:
        label += '  D=' + '{:.3f}'.format(coeff/4) + ' um$^2$/s'
    return label
    
    
