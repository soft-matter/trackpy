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
import diagnostics
import plots
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

def interp(t, pos):
    "Realize a linear interpolation of pos through all t, start to finish."
    first_frame, last_frame = t[[0,-1]]
    full_domain = np.arange(first_frame, 1 + last_frame)
    interpolator = interpolate.interp1d(t, pos, axis=0)
    return full_domain, interpolator(full_domain)

def displacement(x, dt):
    """Return difference between neighbors separated by dt steps (frames).
    This is not the same as numpy.diff(x, n), the nth-order derivative."""
    return x[dt:]-x[:-dt]

def msd(t, pos, mpp, fps, max_interval=100, detail=False):
    """Compute the mean displacement and mean squared displacement of a
    trajectory over a range of time intervals. Input in units of px and frames;
    output in units of microns and seconds."""
    max_interval = min(max_interval, len(t))
    intervals = xrange(1, 1 + max_interval)
    t, pos = interp(t.values, pos.values)
    if detail:
        msd_func = _detailed_msd
        columns = ['t' ,'<x>', '<y>', '<x^2>', '<y^2>', 'msd', 'N']
    else:
        msd_func = _simple_msd
        columns = ['t', 'msd']
    results = [msd_func(t, pos, i, mpp, fps) for i in intervals]
    return DataFrame(np.vstack(results), columns=columns)
     
def _detailed_msd(t, pos, interval, mpp, fps):
    """Given the time points and position points of a trajectory,
    return lag time t, <x>, <y>, <r>, <x^2>, <y^2>, <r^2>, and N."""
    d = displacement(mpp*pos, interval) # [[dx, dy], ...]
    sd = d**2
    stuff = np.column_stack((d, np.sum(d, axis=1), sd, np.sum(sd, axis=1)))
    # [[dx, dy, dr, dx^2, dy^2, dr^2], ...]
    mean_stuff = np.mean(stuff, axis=0)
    # Estimate statistically independent measurements:
    N = np.round(2*stuff.shape[0]/interval)
    return np.append(np.array([interval])/fps, mean_stuff, np.array([N])) 

def _simple_msd(t, pos, interval, mpp, fps):
    """Given the time points and position points of a trajectory, return lag
    time t and mean squared displacment <r^2>."""
    d = displacement(mpp*pos, interval) # [[dx, dy], ...]
    sd = d**2
    msd_result = np.mean(np.sum(sd, axis=1), axis=0)
    return np.array([interval/fps, msd_result]) 

def ensemble_msd(probes, mpp, fps, max_interval=100):
    """Return ensemble mean squared displacement. Input in units of px
    and frames. Output in units of microns and seconds."""
    logger.info("%.3f microns per pixel, %d fps", mpp, fps)
    m = np.vstack([msd(traj, mpp, fps, max_interval, detail=False) \
                for traj in probes])
    m = m[m[:, 0].argsort()] # sort by dt 
    boundaries, = np.where(np.diff(m[:, 0], axis=0) > 0.0)
    boundaries += 1
    m = np.split(m, boundaries) # list of arrays, one for each dt
    ensm_m = np.vstack([np.mean(this_m, axis=0) for this_m in m])
    power, coeff = fit_powerlaw(ensm_m)
    return ensm_m

def fit_powerlaw(a):
    """Fit a power law to MSD data. Return the power and the coefficient.
    This is not a generic power law. By treating it as a linear regression in
    log space, we assume no additive constant: y = 0 + coeff*x**power."""
    slope, intercept, r, p, stderr = \
        stats.linregress(np.log(a[:, 0]), np.log(a[:, 1]))
    return slope, np.exp(intercept)

def compute_drift(traj, suppress_plot=False):
    """Return the ensemble drift, x(t).

    Parameters
    ----------
    traj : a DataFrame that must include columns x, y, frame, and probe

    Returns
    -------
    drift : DataFrame([x, y], index=frame)    
    """
    # Keep only frames 
    delta = pd.concat([t.diff() for p, t in traj.groupby('probe')])
    avg_delta = delta[delta['frame'] == 1].groupby('frame').mean()
    frame, dx = interp(traj['frame'].values, traj[['x', 'y']].values)
    x = np.cumsum(dx, axis=0)
    return DataFrame(x, columns=['x', 'y'], index=frame)

def cart_to_polar(x, y, deg=False):
    "Convert Cartesian x, y to r, theta in radians."
    conversion = 180/pi if deg else 1.
    return np.sqrt(x**2 + y**2), conversion*np.arctan2(y, x)

def subtract_drift(traj, drift=None):
    """Return a copy of probe trajectores with the overall drift subtracted out.
    
    Parameters
    ----------
    traj : a DataFrame that must have columns x, y, and frame
    d : optional DataFrame([x, y], index=frame) like output of drift()
         By default, the drift is calculated from all the probes in traj.
         Optionally specify a different drift, like the drift of a subset
         or superset of these probes.
     

    Returns
    -------
    new_traj : a copy, having modified columns x and y
    """

    if drift is None: 
        logger.info("Computing drift...")
        drift = compute_drift(traj)
    new_traj = traj.copy().set_index('frame', drop=False)
    new_traj[['x', 'y']] -= drift
    print new_traj
    return new_traj

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

def branches(probes):
    """Sort list of probes into three lists, sorted by mobility.
    Return: diffusive, localized, subdiffusive."""
    diffusive = [p for p in probes if is_diffusive(p)]
    localized = [p for p in probes if is_localized(p)]
    subdiffusive = [p for p in probes if ((not is_localized(p)) and \
                           (not is_diffusive(p)))]
    logger.info("{} diffusive, {} localized, {} subdiffusive",
             len(diffusive), len(localized), len(subdiffusive))
    return diffusive, localized, subdiffusive
