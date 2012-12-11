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
import copy
import numpy as np
from scipy import stats
from scipy import interpolate
import pidly
import pandas as pd
from pandas import DataFrame, Series

pi = np.pi

logger = logging.getLogger(__name__)

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

def spline(t, pos, k=3, s=None):
    """Realize a Univariate spline, interpolating pos through all t. 

    Parameters
    ----------
    t : Index (not Series!) of integers with possible gaps
    pos : DataFrame or Series of data to be interpolated
    k : integer
        polynomial order of spline, k <= 5
    s : None or float, optional
        smoothing parameter
    
    Returns
    -------
    DataFrame of interpolated pos. The index is interpolated t.
    """
    first_frame, last_frame = t[[0, -1]]
    domain = np.arange(first_frame, 1 + last_frame)
    new_pos = []
    pos = DataFrame(pos) # in case pos is given as a Series
    for col in pos:
        spl = interpolate.UnivariateSpline(
            t.values, pos[col].values, k=k, s=s)
        new_col = Series(spl(domain), index=domain, name=col)
        new_pos.append(new_col)
    return pd.concat(new_pos, axis=1, keys=pos.columns)

def msd(traj, mpp, fps, max_lagtime=100, detail=False):
    """Compute the mean displacement and mean squared displacement of a
    trajectory over a range of time intervals. Input in units of px and frames;
    output in units of microns and seconds.
    
    Parameters
    ----------
    traj : DataFrame of trajectories, including columns frame, x, and y
        If there is a probe column, the data will be considered probe by probe.
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
    """
    logger.info("%.3f microns per pixel, %d fps", mpp, fps)
    pos = traj[['x', 'y']]
    t = traj['frame']
    # Reindex with consecutive frames, placing NaNs in the gaps. 
    pos = pos.reindex(np.arange(0, t.irow(-1) - t.irow(0) + 1))
    max_lagtime = min(max_lagtime, len(t)) # checking to be safe
    lagtimes = 1 + np.arange(max_lagtime) 
    disp = pd.concat([pos.sub(pos.shift(lt)) for lt in lagtimes],
                     keys=lagtimes, names=['lt', 'frames'])
    results = mpp*disp.mean(level=0)
    results.columns = ['<x>', '<y>']
    results[['<x^2>', '<y^2>']] = mpp**2*(disp**2).mean(level=0)
    results['msd'] = mpp**2*(disp**2).mean(level=0).sum(1) # <r^2>
    # Estimated statistically independent measurements = 2N/t
    if detail:
        results['N'] = 2*disp.icol(0).count(level=0).div(Series(lagtimes))
    results.index = results.index/fps
    return results

def compute_drift(traj, smoothing=None):
    """Return the ensemble drift, x(t).

    Parameters
    ----------
    traj : a DataFrame that must include columns x, y, frame, and probe
    smoothing : float or None, optional
        Positive smoothing factor used to choose the number of knots.
        Number of knots will be increased until the smoothing condition 
        is satisfied:
        sum((w[i]*(y[i]-s(x[i])))**2,axis=0) <= s
        If None (default), s=len(w) which should be a good value if 1/w[i] 
        is an estimate of the standard deviation of y[i]. If 0, spline will
        interpolate through all data points.

    Returns
    -------
    drift : DataFrame([x, y], index=frame)    
    """
    # Probe by probe, take the difference between frames.
    delta = pd.concat([t.set_index('frame', drop=False).diff()
                       for p, t in traj.groupby('probe')])
    # Keep only deltas between frames that are consecutive. 
    delta = delta[delta['frame'] == 1]
    # Restore the original frame column (replacing delta frame).
    delta['frame'] = delta.index
    traj = delta.groupby('frame').mean()
    dx = spline(traj.index, traj[['x', 'y']], s=smoothing)
    x = dx.cumsum(0) 
    return DataFrame(x, columns=['x', 'y'])

def subtract_drift(traj, drift=None):
    """Return a copy of probe trajectores with the overall drift subtracted out.
    
    Parameters
    ----------
    traj : a DataFrame that must have columns x, y, and frame
    drift : optional DataFrame([x, y], index=frame) like output of 
         compute_drift(). If no drift is passed, drift is computed from traj.

    Returns
    -------
    traj : a copy, having modified columns x and y
    """

    if drift is None: 
        drift = compute_drift(traj)
    return traj.set_index('frame', drop=False).sub(drift, fill_value = 0)

def is_localized(traj, threshold=0.4):
    "Is this probe's motion localized?"
    m = msd(traj, mpp=1., fps=1.)
    power, coeff = fit_powerlaw(m)
    return power < threshold

def is_diffusive(traj, threshold=0.85):
    "Is this probe's motion diffusive?"
    m = msd(traj, mpp=1., fps=1.)
    power, coeff = fit_powerlaw(m)
    return power > threshold

def is_smear(traj,threshold):
    d = np.sqrt(np.sum((traj[-1,1:]-traj[0,1:])**2))
    return d > threshold

def is_unphysical(traj, mpp, fps, threshold=0.08):
    """Is the first MSD datapoint unphysically high? (This is sometimes an
    artifact of uneven drift.)"""
    m = msd(traj, mpp, fps=1.)
    return m[0, 1] > threshold
