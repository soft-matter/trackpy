# Copyright 2012 Daniel B. Allan
# dallan@pha.jhu.edu, daniel.b.allan@gmail.com
# http://pha.jhu.edu/~dallan
# http://www.danallan.com
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses>.

from __future__ import division
import logging
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy import interpolate
import pidly

logger = logging.getLogger(__name__)

def msd(traj, mpp, fps, max_lagtime=100, detail=False):
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
    pos = traj[['x', 'y']]
    t = traj['frame']
    # Reindex with consecutive frames, placing NaNs in the gaps. 
    pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1]))
    max_lagtime = min(max_lagtime, len(t)) # checking to be safe
    lagtimes = 1 + np.arange(max_lagtime) 
    disp = pd.concat([pos.sub(pos.shift(lt)) for lt in lagtimes],
                     keys=lagtimes, names=['lagt', 'frames'])
    results = mpp*disp.mean(level=0)
    results.columns = ['<x>', '<y>']
    results[['<x^2>', '<y^2>']] = mpp**2*(disp**2).mean(level=0)
    results['msd'] = mpp**2*(disp**2).mean(level=0).sum(1) # <r^2>
    # Estimated statistically independent measurements = 2N/t
    if detail:
        results['N'] = 2*disp.icol(0).count(level=0).div(Series(lagtimes))
    results['lagt'] = results.index/fps
    return results

def imsd(traj, mpp, fps, max_lagtime=100, statistic='msd'):
    """Compute the mean squared displacements of probes individually.
    
    Parameters
    ----------
    traj : DataFrame of trajectories of multiple probes, including 
        columns probe, frame, x, and y
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
    for pid, ptraj in traj.groupby('probe'):
        msds.append(msd(ptraj, mpp, fps, max_lagtime, False))
        ids.append(pid)
    results = pd.concat(msds, keys=ids)
    # Swap MultiIndex levels so that unstack() makes probes into columns.
    results = results.swaplevel(0, 1)[statistic].unstack()
    return results

def emsd(traj, mpp, fps, max_lagtime=100):
    """Compute the mean squared displacements of an ensemble of probes. 
    
    Parameters
    ----------
    traj : DataFrame of trajectories of multiple probes, including 
        columns probe, frame, x, and y
    mpp : microns per pixel
    fps : frames per second
    max_lagtime : intervals of frames out to which MSD is computed
        Default: 100

    Returns
    -------
    DataFrame([<x>, <y>, <x^2>, <y^2>, msd], index=t)
    
    Notes
    -----
    Input units are pixels and frames. Output units are microns and seconds.
    """
    ids = []
    msds = []
    for pid, ptraj in traj.groupby('probe'):
        msds.append(msd(ptraj, mpp, fps, max_lagtime, True))
        ids.append(pid)
    msds = pd.concat(msds, keys=ids, names=['probe', 'frame'])
    results = msds.mul(msds['N'], axis=0).mean(level=1) # weighted average
    results = results.div(msds['N'].mean(level=1), axis=0) # weights normalized
    # Above, lagt is lumped in with the rest for simplicity and speed.
    # Here, rebuild it from the frame index.
    results['lagt'] = results.index/fps
    del results['N']
    return results

def compute_drift(traj, smoothing=0):
    """Return the ensemble drift, x(t).

    Parameters
    ----------
    traj : DataFrame of trajectories, including columns x, y, frame, and probe
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
    # Probe by probe, take the difference between frames.
    delta = pd.concat([t.set_index('frame', drop=False).diff()
                       for p, t in traj.groupby('probe')])
    # Keep only deltas between frames that are consecutive. 
    delta = delta[delta['frame'] == 1]
    # Restore the original frame column (replacing delta frame).
    delta['frame'] = delta.index
    dx = delta.groupby('frame').mean()
    dx = pd.rolling_mean(dx, smoothing, min_periods=0)
    x = dx.cumsum(0) 
    return DataFrame(x, columns=['x', 'y'])

def subtract_drift(traj, drift=None):
    """Return a copy of probe trajectores with the overall drift subtracted out.
    
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
    return traj.set_index('frame', drop=False).sub(drift, fill_value = 0)

def is_typical(msds, frame=23, lower=0.1, upper=0.9):
    """Examine individual probe MSDs, distinguishing outliers from those
    in the central quantile.

    Parameters
    ----------
    msds : DataFrame like the output of imsd()
        Columns correspond to probes, indexed by lagtime measured in frames.
    frame : integer frame number
        Compare MSDs at this lagtime. Default is 23 (1 second at 24 fps).
    lower : float between 0 and 1, default 0.1
        Probes with MSD up to this quantile are deemed outliers.
    upper : float between 0 and 1, default 0.9
        Probes with MSD above this quantile are deemed outliers.
        
    
    Returns
    -------
    Series of boolean values, indexed by probe number
    True = typical probe, False = outlier probe

    Example
    -------
    m = mr.imsd(traj, MPP, FPS)
    # Index by probe ID, slice using boolean output from is_typical(), and then
    # restore the original index, frame number.
    typical_traj = traj.set_index('probe').ix[is_typical(m)].reset_index()\
        .set_index('frame', drop=False)
    """
    a, b = msds.ix[frame].quantile(lower), msds.ix[frame].quantile(upper)
    return (msds.ix[frame] > a) & (msds.ix[frame] < b)

def idl_track(query, max_disp, min_appearances, memory=3):
    """Call Crocker/Weeks track.pro from IDL using pidly module.
    Returns one big array, where the last column is the probe ID."""
    idl = pidly.IDL()
    logger.info("Opened IDL process.")
    idl('pt = get_sql("{}")'.format(query))
    logger.info("IDL is done loading features from the database. Now tracking....")
    idl('t=track(pt, {}, goodenough={}, memory={})'.format(
        max_disp, min_appearances, memory))
    logger.info("IDL finished tracking. Now piping data into Python....")
    t = idl.ev('t')
    idl.close()
    return DataFrame(
        t, columns=['x', 'y', 'mass', 'size', 'ecc', 'frame', 'probe'])

def vanhove(pos, lagtime=23, mpp=1, ensemble=False, bins=24):
    """Compute the van Hove correlation function at given lagtime (frame span).

    Parameters
    ----------
    pos : DataFrame of x or (or!) y positions, one column per probe, indexed
        by frame
    lagtime : integer interval of frames 
        Compare the correlation function at this lagtime. Default is 23 
        (1 second at 24 fps).
    mpp : microns per pixel, DEFAULT TO 1 because it is usually fine to use
        pixels for this analysis
    ensemble : boolean, defaults False
    bins : integer or sequence
        Specify a number of equally spaced bins, or explicitly specifiy a
        sequence of bin edges. See np.histogram docs.

    Returns
    -------
    vh : If ensemble=True, a DataFrame with each probe's van Hove correlation 
        function, indexed by displacement. If ensemble=False, a Series with 
        the van Hove correlation function of the whole ensemble.

    Example
    -------
    pos = traj.set_index(['frame', 'probe'])['x'].unstack() # probes as columns
    vh = vanhove(pos)
    """
    # Reindex with consecutive frames, placing NaNs in the gaps. 
    pos = pos.reindex(np.arange(pos.index[0], 1 + pos.index[-1]))
    assert lagtime <= pos.index.values.max(), \
        "There is a no data out to frame %s. " % frame
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

def is_not_dirt(traj, threshold=3, mpp=1):
    """Identify which probes are so localized that they are probably dirt.
    
    Parameters
    ----------
    traj : DataFrame of trajectories of multiple probes, including 
        columns probe, frame, x, and y
    threshold : minimum displacement of non-dirt
    mpp : microns per pixel, assumed to be 1

    Returns
    -------
    boolean Series indexed by probe ID. False = dirt.

    Notes
    -----
    Use this before you subtract the overall drift, not after.

    Example
    -------
    >>> notdirt = is_not_dirt(t)
    >>> t.set_index('probe').ix[notdirt].reset_index().set_index('frame', drop=False)
    """
    
    extremes = traj.groupby('probe')['x', 'y'].agg(['max', 'min'])
    diag_size = np.sqrt((extremes[('x', 'max')] - extremes[('x', 'min')])**2
                        + (extremes[('y', 'max')] - extremes[('y', 'min')])**2)
    return diag_size > threshold 

def is_localized(traj, threshold=0.4):
    raise NotImplementedError, "I will rewrite this."

def is_diffusive(traj, threshold=0.9):
    raise NotImplementedError, "I will rewrite this."
