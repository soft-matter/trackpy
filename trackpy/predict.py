# Copyright 2014, Nathan C. Keim
# keimnathan@gmail.com

"""Tools to improve tracking performance by guessing where a particle will appear next."""

from warnings import warn
from collections import deque
import functools
import numpy as np
from scipy.interpolate import NearestNDInterpolator, interp1d
import pandas as pd
from . import linking

def predictor(predict_func):
    """Decorator to vectorize a predictor function for a single particle.

    Converts P(t1, particle) into Pvec(t1, particles), where 'particles' is a list
    of particle instances, and 'Pvec' can be passed to a linking function
    (e.g. link_df_iter()) via its 'predictor' argument.
    """
    def Pvec(t1, particles):
        targeted_p = functools.partial(predict_func, t1)
        return map(targeted_p, particles)
    return Pvec

@predictor
def null_predict(t1, particle):
    return (particle.pos)

class NullPredict(object):
    def link_df_iter(self, *args, **kw):
        """Wrapper for linking.link_df_iter() that causes it to use this predictor."""
        if getattr(self, '_already_linked', False):
            warn('Perform tracking with a fresh predictor instance to avoid surprises.')
        self._already_linked = True
        kw['predictor'] = self.predict
        self.pos_columns = kw.get('pos_columns', ['x', 'y'])
        self.t_column = kw.get('t_column', 'frame')
        for frame in linking.link_df_iter(*args, **kw):
            self.observe(frame)
            yield frame
    def observe(self, frame):
        """Examine the latest output of the linker, to update our predictions."""
        pass
    def predict(self, t1, particles):
        """Predict the positions of 'particles' at time 't1'"""
        return map(lambda p: p.pos, particles)

class _RecentVelocityPredict(NullPredict):
    def __init__(self):
        # Use the last 2 frames to make a velocity field
        self.recent_frames = deque([], 2)
    def _compute_velocities(self, frame):
        """Compute velocity field based on a newly-tracked frame."""
        pframe = frame.set_index('particle')
        self.recent_frames.append(pframe)
        if len(self.recent_frames) == 1:
            # Double the first frame. Velocity field will be zero.
            self.recent_frames.append(pframe)
            dt = 1. # Avoid dividing by zero
        else: # Not the first frame
            dt = self.recent_frames[1][self.t_column].values[0] - \
                 self.recent_frames[0][self.t_column].values[0]

        # Compute velocity field
        disps = self.recent_frames[1][self.pos_columns].join(
            self.recent_frames[1][self.pos_columns] - \
                self.recent_frames[0][self.pos_columns], rsuffix='_disp_').dropna()
        positions = disps[self.pos_columns]
        vels = disps[[cn + '_disp_' for cn in self.pos_columns]] / dt
        # 'vels' will have same column names as 'positions'
        vels.rename(columns=lambda n: n[:-6], inplace=True)
        return dt, positions, vels

class NearestVelocityPredict(_RecentVelocityPredict):
    """Predict a particle's position based on the most recent nearby velocity.

    Parameters
    ----------
    initial_guess_positions : Nxd array, optional
    initial_guess_vels : Nxd array, optional
        If specified, these initialize the velocity field with velocity
        samples at the given points.
    """

    def __init__(self, initial_guess_positions=None,
                 initial_guess_vels=None):
        super(NearestVelocityPredict, self).__init__()
        if initial_guess_positions is not None:
            self.use_initial_guess = True
            self.interpolator = NearestNDInterpolator(
                np.asarray(initial_guess_positions),
                np.asarray(initial_guess_vels))
        else:
            self.use_initial_guess = False
    def observe(self, frame):
        dt, positions, vels = self._compute_velocities(frame)
        if self.use_initial_guess:
            self.use_initial_guess = False
        else:
            if positions.values.shape[0] > 0:
                self.interpolator = NearestNDInterpolator(positions.values, vels.values)
            else:
                # Sadly, the 2 most recent frames had no points in common.
                warn('Could not generate velocity field for prediction: no tracks')
                def null_interpolator(*x):
                    return np.zeros((len(x),))
                self.interpolator = null_interpolator
    def predict(self, t1, particles):
        poslist, tlist = zip(*[(p.pos, p.t) for p in particles])
        positions = np.array(poslist)
        times = np.array(tlist)
        return positions + self.interpolator(positions) * \
               np.tile(t1 - times, (positions.shape[1], 1)).T

class DriftPredict(_RecentVelocityPredict):
    """Predict a particle's position based on the mean velocity of all particles.

    Parameters
    ----------
    initial_guess : Array of length d. Otherwise assumed to be zero velocity.
    """
    def __init__(self, initial_guess=None):
        super(DriftPredict, self).__init__()
        self.initial_guess = initial_guess
    def observe(self, frame):
        dt, positions, vels = self._compute_velocities(frame)
        if self.initial_guess is not None:
            self.vel = np.asarray(self.initial_guess)
            self.initial_guess = None
        else:
            self.vel = vels.mean().values
    def predict(self, t1, particles):
        poslist, tlist = zip(*[(p.pos, p.t) for p in particles])
        positions = np.array(poslist)
        times = np.array(tlist)
        return positions + self.vel * \
               np.tile(t1 - times, (positions.shape[1], 1)).T

class ChannelPredict(_RecentVelocityPredict):
    """Predict a particle's position based on its spanwise coordinate in a channel.

    This operates by binning particles according to their spanwise coordinate and
    averaging velocity, to make an instantaneous velocity profile.

    Parameters
    ----------
    bin_size : Size of bins, in units of spanwise length, over which to average
        streamwise velocity.
    flow_axis : Name of coordinate along which particles are flowing (default "x")
    minsamples : Minimum number of particles in a bin for its average
        velocity to be valid.
    initial_profile_guess : Nx2 array (optional)
        (spanwise coordinate, streamwise velocity) samples specifying
        initial velocity profile. Samples must be sufficiently dense to account
        for variation in the velocity profile. If omitted, initial velocities are
        assumed to be zero.

    Notes
    -----
    - This currently only works for 2D data.
    - Where there were not enough data to make an average velocity (N < minsamples),
        we borrow from the nearest valid bin.
    """
    def __init__(self, bin_size, flow_axis='x', minsamples=20,
                 initial_profile_guess=None):
        super(ChannelPredict, self).__init__()
        self.bin_size = bin_size
        self.flow_axis = flow_axis
        self.minsamples = minsamples
        # Use the last 2 frames to make a velocity field
        self.recent_frames = deque([], 2)
        self.initial_profile_guess = initial_profile_guess
    def observe(self, frame):
        # Sort out dimensions and axes
        if len(self.pos_columns) != 2:
            raise ValueError('Implemented for 2 dimensions only')
        if self.flow_axis not in self.pos_columns:
            raise ValueError('pos_columns (%r) does not include the specified flow_axis (%s)!' % \
                             (self.pos_columns, self.flow_axis))
        poscols = self.pos_columns[:]
        flow_axis_position = poscols.index(self.flow_axis)
        poscols.remove(self.flow_axis)
        span_axis = poscols[0]

        # Make velocity profile
        dt, positions, vels = self._compute_velocities(frame)

        if self.initial_profile_guess is not None:
            self.initial_profile_guess = np.asarray(self.initial_profile_guess)
            prof = pd.Series(self.initial_profile_guess[:,1],
                                 index=self.initial_profile_guess[:,0])
        else:
            # Bin centers
            vels['bin'] = positions[span_axis] - positions[span_axis] \
                                                 % self.bin_size + self.bin_size / 2.
            grpvels = vels.groupby('bin')[self.flow_axis]
            # Only use bins that have enough samples
            profcount = grpvels.count()
            prof = grpvels.mean()[profcount >= self.minsamples]

        if len(prof) > 0:
            # Handle boundary conditions for interpolator
            prof_ind, prof_vals = list(prof.index), list(prof)
            prof_ind.insert(0, -np.inf)
            prof_ind.append(np.inf)
            prof_vals.insert(0, prof.values[0])
            prof_vals.append(prof.values[-1])
            prof_vels = pd.DataFrame({self.flow_axis: pd.Series(prof_vals, index=prof_ind),
                                      span_axis: 0})
            prof_interp = interp1d(prof_vels.index.values, prof_vels[self.pos_columns].values,
                                   'nearest', axis=0)
            if flow_axis_position == 0:
                self.interpolator = lambda x: prof_interp(x[:,1])
            else:
                self.interpolator = lambda x: prof_interp(x[:,0])
        else:
            # Not enough samples in any bin
            warn('Could not generate velocity field for prediction: ' + \
                 'not enough tracks or bin_size too small')
            nullvel = np.zeros((len(self.pos_columns),))
            def null_interpolator(x):
                return nullvel
            self.interpolator = null_interpolator
    def predict(self, t1, particles):
        poslist, tlist = zip(*[(p.pos, p.t) for p in particles])
        positions = np.array(poslist)
        times = np.array(tlist)
        return positions + self.interpolator(positions) * \
               np.tile(t1 - times, (positions.shape[1], 1)).T



