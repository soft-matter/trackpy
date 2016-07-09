from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import warnings
from warnings import warn
from .utils import pandas_sort


def msd(traj, mpp, fps, max_lagtime=100, detail=False, pos_columns=None):
    """Compute the mean displacement and mean squared displacement of one
    trajectory over a range of time intervals.

    Parameters
    ----------
    traj : DataFrame with one trajectory, including columns frame, x, and y
    mpp : microns per pixel
    fps : frames per second
    max_lagtime : intervals of frames out to which MSD is computed
        Default: 100
    detail : See below. Default False.

    Returns
    -------
    DataFrame([<x>, <y>, <x^2>, <y^2>, msd], index=t)

    If detail is True, the DataFrame also contains a column N,
    the estimated number of statistically independent measurements
    that comprise the result at each lagtime.

    Notes
    -----
    Input units are pixels and frames. Output units are microns and seconds.

    See also
    --------
    imsd() and emsd()
    """
    if traj['frame'].max() - traj['frame'].min() + 1 == len(traj):
        # no gaps: use fourier-transform algorithm
        return _msd_fft(traj, mpp, fps, max_lagtime, detail, pos_columns)
    else:
        # there are gaps in the trajectory: use slower algorithm
        return _msd_gaps(traj, mpp, fps, max_lagtime, detail, pos_columns)


def _msd_N(N, t):
    """Computes the effective number of statistically independent measurements
    of the mean square displacement of a single trajectory.

    Parameters
    ----------
    N : integer
        the number of positions in the trajectory (=number of steps + 1)
    t : iterable
        an iterable of lagtimes (integers)

    References
    ----------
    Derived from Equation B4 in:
    Qian, Hong, Michael P. Sheetz, and Elliot L. Elson. "Single particle
    tracking. Analysis of diffusion and flow in two-dimensional systems."
    Biophysical journal 60.4 (1991): 910.
    """
    t = np.array(t, dtype=np.float)
    return np.where(t > N/2,
                    1/(1+((N-t)**3+5*t-4*(N-t)**2*t-N)/(6*(N-t)*t**2)),
                    6*(N-t)**2*t/(2*N-t+4*N*t**2-5*t**3))


def _msd_iter(pos, lagtimes):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for lt in lagtimes:
            diff = pos[lt:] - pos[:-lt]
            yield np.concatenate((np.nanmean(diff, axis=0),
                                  np.nanmean(diff**2, axis=0)))



def _msd_gaps(traj, mpp, fps, max_lagtime=100, detail=False, pos_columns=None):
    """Compute the mean displacement and mean squared displacement of one
    trajectory over a range of time intervals."""
    if pos_columns is None:
        pos_columns = ['x', 'y']
    result_columns = ['<{}>'.format(p) for p in pos_columns] + \
                     ['<{}^2>'.format(p) for p in pos_columns]

    # Reindex with consecutive frames, placing NaNs in the gaps.
    pos = traj.set_index('frame')[pos_columns] * mpp
    pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1]))

    max_lagtime = min(max_lagtime, len(pos) - 1)  # checking to be safe

    lagtimes = np.arange(1, max_lagtime + 1)

    result = pd.DataFrame(_msd_iter(pos.values, lagtimes),
                          columns=result_columns, index=lagtimes)
    result['msd'] = result[result_columns[-len(pos_columns):]].sum(1)
    if detail:
        # effective number of measurements
        # approximately corrected with number of gaps
        result['N'] = _msd_N(len(pos), lagtimes) * len(traj) / len(pos)
    result['lagt'] = result.index.values/float(fps)
    result.index.name = 'lagt'
    return result


def _msd_fft(traj, mpp, fps, max_lagtime=100, detail=False, pos_columns=None):
    """Compute the mean displacement and mean squared displacement of one
    trajectory over a range of time intervals using FFT transformation.

    The original Python implementation comes from a SO answer :
    http://stackoverflow.com/questions/34222272/computing-mean-square-displacement-using-python-and-fft#34222273.
    The algorithm is described in this paper : http://dx.doi.org/10.1051/sfn/201112010.
    """
    if pos_columns is None:
        pos_columns = ['x', 'y']
    result_columns = ['<{}>'.format(p) for p in pos_columns] + \
                     ['<{}^2>'.format(p) for p in pos_columns]

    r = traj[pos_columns].values * mpp
    t = traj['frame']

    max_lagtime = min(max_lagtime, len(t) - 1)  # checking to be safe
    lagtimes = np.arange(1, max_lagtime + 1)
    N = len(r)

    # calculate the mean displacements
    r_diff = r[:-max_lagtime-1:-1] - r[:max_lagtime]
    disp = np.cumsum(r_diff, axis=0) / (N - lagtimes[:, np.newaxis])

    # below is a vectorized version of the original code
    D = r**2
    D_sum = D[:max_lagtime] + D[:-max_lagtime-1:-1]
    S1 = 2*D.sum(axis=0) - np.cumsum(D_sum, axis=0)
    F = np.fft.fft(r, n=2*N, axis=0)  # 2*N because of zero-padding
    PSD = F * F.conjugate()
    # this is the autocorrelation in convention B:
    S2 = np.fft.ifft(PSD, axis=0)[1:max_lagtime+1].real
    squared_disp = S1 - 2 * S2
    squared_disp /= N - lagtimes[:, np.newaxis]  # divide res(m) by (N-m)

    results = pd.DataFrame(np.concatenate((disp, squared_disp), axis=1),
                           index=lagtimes, columns=result_columns)
    results['msd'] = squared_disp.sum(axis=1)
    if detail:
        results['N'] = _msd_N(N, lagtimes)
    results['lagt'] = lagtimes / float(fps)
    results.index.name = 'lagt'

    return results


def imsd(traj, mpp, fps, max_lagtime=100, statistic='msd', pos_columns=None):
    """Compute the mean squared displacement of each particle.

    Parameters
    ----------
    traj : DataFrame of trajectories of multiple particles, including
        columns particle, frame, x, and y
    mpp : microns per pixel
    fps : frames per second
    max_lagtime : intervals of frames out to which MSD is computed
        Default: 100
    statistic : {'msd', '<x>', '<y>', '<x^2>', '<y^2>'}, default is 'msd'
        The functions msd() and emsd() return all these as columns. For
        imsd() you have to pick one.

    Returns
    -------
    DataFrame([Probe 1 msd, Probe 2 msd, ...], index=t)

    Notes
    -----
    Input units are pixels and frames. Output units are microns and seconds.
    """
    ids = []
    msds = []
    # Note: Index is set by msd, so we don't need to worry
    # about conformity here.
    for pid, ptraj in traj.groupby('particle'):
        msds.append(msd(ptraj, mpp, fps, max_lagtime, False, pos_columns))
        ids.append(pid)
    results = pd.concat(msds, keys=ids)
    # Swap MultiIndex levels so that unstack() makes particles into columns.
    results = results.swaplevel(0, 1)[statistic].unstack()
    lagt = results.index.values.astype('float64')/float(fps)
    results.set_index(lagt, inplace=True)
    results.index.name = 'lag time [s]'
    return results


def emsd(traj, mpp, fps, max_lagtime=100, detail=False, pos_columns=None):
    """Compute the ensemble mean squared displacements of many particles.

    Parameters
    ----------
    traj : DataFrame of trajectories of multiple particles, including
        columns particle, frame, x, and y
    mpp : microns per pixel
    fps : frames per second
    max_lagtime : intervals of frames out to which MSD is computed
        Default: 100
    detail : Set to True to include <x>, <y>, <x^2>, <y^2>. Returns
        only <r^2> by default.

    Returns
    -------
    Series[msd, index=t] or, if detail=True,
    DataFrame([<x>, <y>, <x^2>, <y^2>, msd], index=t)

    Notes
    -----
    Input units are pixels and frames. Output units are microns and seconds.
    """
    ids = []
    msds = []
    for pid, ptraj in traj.reset_index(drop=True).groupby('particle'):
        msds.append(msd(ptraj, mpp, fps, max_lagtime, True, pos_columns))
        ids.append(pid)
    msds = pd.concat(msds, keys=ids, names=['particle', 'frame'])
    results = msds.mul(msds['N'], axis=0).mean(level=1)  # weighted average
    results = results.div(msds['N'].mean(level=1), axis=0)  # weights normalized
    # Above, lagt is lumped in with the rest for simplicity and speed.
    # Here, rebuild it from the frame index.
    if not detail:
        return results.set_index('lagt')['msd']
    return results


def compute_drift(traj, smoothing=0, pos_columns=None):
    """Return the ensemble drift, x(t).

    Parameters
    ----------
    traj : DataFrame of trajectories, including columns x, y, frame, and particle
    smoothing : integer
        Smooth the drift using a forward-looking rolling mean over
        this many frames.

    Returns
    -------
    drift : DataFrame([x, y], index=frame)

    Examples
    --------
    >>> compute_drift(traj).plot() # Default smoothing usually smooths too much.
    >>> compute_drift(traj, 0).plot() # not smoothed
    >>> compute_drift(traj, 15).plot() # Try various smoothing values.
    >>> drift = compute_drift(traj, 15) # Save good drift curves.
    >>> corrected_traj = subtract_drift(traj, drift) # Apply them.
    """
    if pos_columns is None:
        pos_columns = ['x', 'y']
    # Probe by particle, take the difference between frames.
    delta = traj.groupby('particle').apply(lambda x :
                                    x.set_index('frame', drop=False).diff())
    # Keep only deltas between frames that are consecutive.
    delta = delta[delta['frame'] == 1]
    # Restore the original frame column (replacing delta frame).
    del delta['frame']
    delta.reset_index('particle', drop=True, inplace=True)
    delta.reset_index('frame', drop=False, inplace=True)
    dx = delta.groupby('frame').mean()
    if smoothing > 0:
        dx = pd.rolling_mean(dx, smoothing, min_periods=0)
    x = dx.cumsum(0)[pos_columns]
    return x


def subtract_drift(traj, drift=None, inplace=False):
    """Return a copy of particle trajectories with the overall drift subtracted
    out.

    Parameters
    ----------
    traj : DataFrame of trajectories, including columns x, y, and frame
    drift : optional DataFrame([x, y], index=frame) like output of
         compute_drift(). If no drift is passed, drift is computed from traj.

    Returns
    -------
    traj : a copy, having modified columns x and y
    """
    if drift is None:
        drift = compute_drift(traj)
    if not inplace:
        traj = traj.copy()
    traj.set_index('frame', inplace=True, drop=False)
    traj.sort_index(inplace=True)
    for col in drift.columns:
        traj[col] = traj[col].sub(drift[col], fill_value=0, level='frame')
    return traj


def is_typical(msds, frame, lower=0.1, upper=0.9):
    """Identify which paritcles' MSDs are in the central quantile.

    Parameters
    ----------
    msds : DataFrame
        This should be organized like the output of imsd().
        Columns correspond to particles, indexed by lagtime in frames.
    frame : integer
        Compare MSDs at this lag interval.
    lower : float between 0 and 1, default 0.1
        Probes with MSD up to this quantile are deemed outliers.
    upper : float between 0 and 1, default 0.9
        Probes with MSD above this quantile are deemed outliers.

    Returns
    -------
    Series of boolean values, indexed by particle number
    True = typical particle, False = outlier particle

    Examples
    --------

    >>> m = tp.imsd(traj, MPP, FPS)
    >>> # Index by particle ID, slice using boolean output from is_typical(), and then
    >>> # restore the original index, frame number.
    >>> typical_traj = traj.set_index('particle').ix[is_typical(m)]\
    .reset_index().set_index('frame', drop=False)
    """
    a, b = msds.iloc[frame].quantile(lower), msds.iloc[frame].quantile(upper)
    return (msds.iloc[frame] > a) & (msds.iloc[frame] < b)


def vanhove(pos, lagtime, mpp=1, ensemble=False, bins=24):
    """Compute the van Hove correlation (histogram of displacements).

    The van Hove correlation function is simply a histogram of particle
    displacements. It is useful for detecting physical heterogeneity
    (or tracking errors).

    Parameters
    ----------
    pos : DataFrame
        x or (or!) y positions, one column per particle, indexed by frame
    lagtime : integer interval of frames
        Compare the correlation function at this lagtime.
    mpp : microns per pixel, DEFAULT TO 1 because it is usually fine to use
        pixels for this analysis
    ensemble : boolean, defaults False
    bins : integer or sequence
        Specify a number of equally spaced bins, or explicitly specifiy a
        sequence of bin edges. See np.histogram docs.

    Returns
    -------
    vh : DataFrame or Series
        If ensemble=True, a DataFrame with each particle's van Hove correlation
        function, indexed by displacement. If ensemble=False, a Series with
        the van Hove correlation function of the whole ensemble.

    Examples
    --------
    >>> pos = traj.set_index(['frame', 'particle'])['x'].unstack() # particles as columns
    >>> vh = vanhove(pos)
    """
    # Reindex with consecutive frames, placing NaNs in the gaps.
    pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1]))
    assert lagtime <= pos.index.values.max(), \
        "There is a no data out to frame %s. " % pos.index.values.max()
    disp = mpp*pos.sub(pos.shift(lagtime))
    # Let np.histogram choose the best bins for all the data together.
    values = disp.values.flatten()
    values = values[np.isfinite(values)]
    global_bins = np.histogram(values, bins=bins)[1]
    # Use those bins to histogram each column by itself.
    vh = disp.apply(
        lambda x: Series(np.histogram(x, bins=global_bins, density=True)[0]))
    vh.index = global_bins[:-1]
    if ensemble:
        return vh.sum(1)/len(vh.columns)
    else:
        return vh


def diagonal_size(single_trajectory, pos_columns=None, t_column='frame'):
    """Measure the diagonal size of a trajectory.

    Parameters
    ----------
    single_trajectory : DataFrame containing a single trajectory
    pos_columns = list
        names of column with position ['x', 'y']
    t_column = 'frame'

    Returns
    -------
    float : length of diangonal of rectangular box containing the trajectory

    Examples
    --------
    >>> diagonal_size(single_trajectory)

    >>> many_trajectories.groupby('particle').agg(tp.diagonal_size)

    >>> many_trajectories.groupby('particle').filter(lambda x: tp.diagonal_size(x) > 5)
    """
    if pos_columns is None:
        pos_columns = ['x', 'y']
    pos = single_trajectory.set_index(t_column)[pos_columns]
    return np.sqrt(np.sum(pos.apply(np.ptp)**2))


def is_localized(traj, threshold=0.4):
    raise NotImplementedError("This function has been removed.")


def is_diffusive(traj, threshold=0.9):
    raise NotImplementedError("This function has been removed.")


def relate_frames(t, frame1, frame2, pos_columns=None):
    """Find the displacement vector of all particles between two frames.

    Parameters
    ----------
    t : DataFrame
        trajectories
    pos_columns = list
        names of column with position ['x', 'y']
    frame1 : integer
    frame2 : integer

    Returns
    -------
    DataFrame
        indexed by particle, containing:
        x, y, etc. (corresponding to frame1)
        x_b, y_b, etc. (corresponding to frame2)
        dx, dy, etc.
        dr
        direction (only if pos_columns=['x', 'y'])
    """
    if pos_columns is None:
        pos_columns = ['x', 'y']
    a = t[t.frame == frame1]
    b = t[t.frame == frame2]
    j = a.set_index('particle')[pos_columns].join(
         b.set_index('particle')[pos_columns], rsuffix='_b')
    for pos in pos_columns:
        j['d' + pos] = j[pos + '_b'] - j[pos]
    j['dr'] = np.sqrt(np.sum([j['d' + pos]**2 for pos in pos_columns], 0))
    if pos_columns == ['x', 'y']:
        j['direction'] = np.arctan2(j.dy, j.dx)
    return j


def direction_corr(t, frame1, frame2):
    """Compute the cosine between every pair of particles' displacements.

    Parameters
    ----------
    t : DataFrame
        trajectories, containing columns particle, frame, x, and y
    frame1 : frame number
    frame2 : frame number

    Returns
    -------
    DataFrame, indexed by particle, including dx, dy, and direction
    """
    j = relate_frames(t, frame1, frame2)
    cosine = np.cos(np.subtract.outer(j.direction, j.direction))
    r = np.sqrt(np.subtract.outer(j.x, j.x)**2 +
                np.subtract.outer(j.y, j.y)**2)
    upper_triangle = np.triu_indices_from(r, 1)
    result = DataFrame({'r': r[upper_triangle],
                        'cos': cosine[upper_triangle]})
    return result


def velocity_corr(t, frame1, frame2):
    """Compute the velocity correlation between
    every pair of particles' displacements.

    Parameters
    ----------
    t : DataFrame
        trajectories, containing columns particle, frame, x, and y
    frame1 : frame number
    frame2 : frame number

    Returns
    -------
    DataFrame, indexed by particle, including dx, dy, and direction
    """
    j = relate_frames(t, frame1, frame2)
    cosine = np.cos(np.subtract.outer(j.direction, j.direction))
    r = np.sqrt(np.subtract.outer(j.x, j.x)**2 +
                np.subtract.outer(j.y, j.y)**2)
    dot_product = cosine*np.abs(np.multiply.outer(j.dr, j.dr))
    upper_triangle = np.triu_indices_from(r, 1)
    result = DataFrame({'r': r[upper_triangle],
                        'dot_product': dot_product[upper_triangle]})
    return result


def theta_entropy(pos, bins=24, plot=True):
    """Plot the distribution of directions and return its Shannon entropy.

    Parameters
    ----------
    pos : DataFrame with columns x and y, indexed by frame
    bins : number of equally-spaced bins in distribution. Default 24.
    plot : plot direction histogram if True

    Returns
    -------
    float : Shannon entropy

    Examples
    --------
    >>> theta_entropy(t[t['particle'] == 3].set_index('frame'))

    >>> S = t.set_index('frame').groupby('particle').apply(tp.theta_entropy)
    """

    disp = pos - pos.shift(1)
    direction = np.arctan2(disp['y'], disp['x'])
    bins = np.linspace(-np.pi, np.pi, bins + 1)
    if plot:
        Series(direction).hist(bins=bins)
    return shannon_entropy(direction.dropna(), bins)


def shannon_entropy(x, bins):
    """Compute the Shannon entropy of the distribution of x."""
    hist = np.histogram(x, bins)[0]
    hist = hist.astype('float64')/hist.sum()  # normalize probablity dist.
    entropy = -np.sum(np.nan_to_num(hist*np.log(hist)))
    return entropy


def min_rolling_theta_entropy(pos, window=24, bins=24):
    """Compute the minimum Shannon entropy in any window.

    Parameters
    ----------
    pos : DataFrame with columns x and y, indexed by frame
    window : number of observations per window
    bins : number of equally-spaced bins in distribution. Default 24.

    Returns
    -------
    float : Shannon entropy

    Examples
    --------
    >>> theta_entropy(t[t['particle'] == 3].set_index('frame'))

    >>> S = t.set_index('frame').groupby('particle').apply(
    ...     tp.min_rolling_theta_entropy)
    """

    disp = pos - pos.shift(1)
    direction = np.arctan2(disp['y'], disp['x'])
    bins = np.linspace(-np.pi, np.pi, bins + 1)
    f = lambda x: shannon_entropy(x, bins)
    return pd.rolling_apply(direction.dropna(), window, f).min()


def proximity(*args, **kwargs):
    warn('This function has been moved to `trackpy.static', DeprecationWarning)
    from trackpy.static import proximity
    return proximity(*args, **kwargs)
