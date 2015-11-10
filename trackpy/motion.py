from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.spatial import cKDTree

def msd(traj, mpp, fps, max_lagtime=100, detail=False, pos_columns=['x', 'y']):
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
    pos = traj.set_index('frame')[pos_columns]
    t = traj['frame']
    # Reindex with consecutive frames, placing NaNs in the gaps.
    pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1]))
    max_lagtime = min(max_lagtime, len(t))  # checking to be safe
    lagtimes = 1 + np.arange(max_lagtime)
    disp = pd.concat([pos.sub(pos.shift(lt)) for lt in lagtimes],
                     keys=lagtimes, names=['lagt', 'frames'])
    results = mpp*disp.mean(level=0)
    results.columns = ['<{}>'.format(p) for p in pos_columns]
    results[['<{}^2>'.format(p) for p in pos_columns]] = mpp**2*(disp**2).mean(level=0)
    results['msd'] = mpp**2*(disp**2).mean(level=0).sum(1) # <r^2>
    # Estimated statistically independent measurements = 2N/t
    if detail:
        results['N'] = 2*disp.iloc[:,0].count(level=0).div(Series(lagtimes))
    results['lagt'] = results.index.values/fps
    return results[:-1]


def imsd(traj, mpp, fps, max_lagtime=100, statistic='msd', pos_columns=['x', 'y']):
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


def emsd(traj, mpp, fps, max_lagtime=100, detail=False, pos_columns=['x', 'y']):
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


def compute_drift(traj, smoothing=0, pos_columns=['x', 'y']):
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
    compute_drift(traj).plot() # Default smoothing usually smooths too much.
    compute_drift(traj, 0).plot() # not smoothed
    compute_drift(traj, 15).plot() # Try various smoothing values.

    drift = compute_drift(traj, 15) # Save good drift curves.
    corrected_traj = subtract_drift(traj, drift) # Apply them.
    """
    # Probe by particle, take the difference between frames.
    delta = pd.concat([t.set_index('frame', drop=False).diff()
                       for p, t in traj.groupby('particle')])
    # Keep only deltas between frames that are consecutive.
    delta = delta[delta['frame'] == 1]
    # Restore the original frame column (replacing delta frame).
    del delta['frame']
    delta.reset_index(inplace=True)
    dx = delta.groupby('frame').mean()
    if smoothing > 0:
        dx = pd.rolling_mean(dx, smoothing, min_periods=0)
    x = dx.cumsum(0)[pos_columns]
    return x


def subtract_drift(traj, drift=None):
    """Return a copy of particle trajectores with the overall drift subtracted out.

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
    return traj.set_index('frame', drop=False).sub(drift, fill_value=0)


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

    m = tp.imsd(traj, MPP, FPS)
    # Index by particle ID, slice using boolean output from is_typical(), and then
    # restore the original index, frame number.
    typical_traj = traj.set_index('particle').ix[is_typical(m)].reset_index()\
        .set_index('frame', drop=False)
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
    pos = traj.set_index(['frame', 'particle'])['x'].unstack() # particles as columns
    vh = vanhove(pos)
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
    """Plot the distrbution of directions and return its Shannon entropy.

    Parameters
    ----------
    pos : DataFrame with columns x and y, indexed by frame
    bins : number of equally-spaced bins in distribution. Default 24.
    plot : plot direction historgram if True

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


def proximity(features, pos_columns=None):
    """Find the distance to each feature's nearest neighbor.

    Parameters
    ----------
    features : DataFrame
    pos_columns : list of column names
        ['x', 'y'] by default

    Returns
    -------
    proximity : DataFrame
        distance to each particle's nearest neighbor,
        indexed by particle if 'particle' column is present in input

    Examples
    --------
    Find the proximity of each particle to its nearest neighbor in every frame.

    >>> prox = t.groupby('frame').apply(proximity).reset_index()
    >>> avg_prox = prox.groupby('particle')['proximity'].mean()

    And filter the trajectories...

    >>> particle_nos = avg_prox[avg_prox > 20].index
    >>> t_filtered = t[t['particle'].isin(particle_nos)]
    """
    if pos_columns is None:
        pos_columns = ['x', 'y']
    leaf_size = max(1, int(np.round(np.log10(len(features)))))
    tree = cKDTree(features[pos_columns].copy(), leaf_size)
    proximity = tree.query(tree.data, 2)[0][:, 1]
    result = DataFrame({'proximity': proximity})
    if 'particle' in features:
        result.set_index(features['particle'], inplace=True)
    return result
