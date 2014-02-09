# Copyright 2014, Nathan C. Keim
# keimnathan@gmail.com

"""Tools to improve tracking performance by guessing where a particle will appear next."""

from collections import deque
import numpy as np
from scipy.interpolate import NearestNDInterpolator, interp1d
import pandas as pd
from . import linking

def null_predict(particle, source_frame, dest_frame):
    return (particle.pos)

class NullPredict(object):
    def link_df_iter(self, *args, **kw):
        """Wrapper for linking.link_df_iter() that causes it to use this predictor."""
        if getattr(self, '_already_linked', False):
            raise UserWarning('Perform tracking with a fresh predictor instance to avoid surprises.')
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
    def predict(self, t1, particle):
        """Predict the position of 'particle' at time 't1'"""
        return particle.pos

class NearestVelocityPredict(NullPredict):
    """Predict a particle's position based on the most recent nearby velocity.

    The guess for the first frame is zero velocity.
    """
    def __init__(self):
        # Use the last 2 frames to make a velocity field
        self.recent_frames = deque([], 2)
    def observe(self, frame):
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
        positions = disps[self.pos_columns].values
        vels = disps[[cn + '_disp_' for cn in self.pos_columns]].values / dt
        self.ndims = len(self.pos_columns)
        if positions.shape[0] > 0:
            self.interpolator = NearestNDInterpolator(positions, vels)
        else:
            # Sadly, the 2 most recent frames had no points in common.
            raise UserWarning('Could not generate velocity field for prediction: no tracks')
            def null_interpolator(*x):
                return np.zeros((len(x),))
            self.interpolator = null_interpolator
    def predict(self, t1, particle):
        prediction = particle.pos +  \
                     self.interpolator(*particle.pos) * (t1 - particle.t)
        #print t1, particle, prediction
        return prediction

class ChannelPredict(NullPredict):
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

    Notes
    -----
    - This currently only works for 2D data.
    - The guess for the first frame is zero velocity.
    - Where there were not enough data to make an average velocity (N < minsamples),
        we borrow from the nearest valid bin.
    """
    def __init__(self, bin_size, flow_axis='x', minsamples=20):
        self.bin_size = bin_size
        self.flow_axis = flow_axis
        self.minsamples = minsamples
        # Use the last 2 frames to make a velocity field
        self.recent_frames = deque([], 2)
    def observe(self, frame):
        pframe = frame.set_index('particle')
        self.recent_frames.append(pframe)
        if len(self.recent_frames) == 1:
            # Double the first frame. Velocity field will be zero.
            self.recent_frames.append(pframe)
            dt = 1. # Avoid dividing by zero
        else: # Not the first frame
            dt = self.recent_frames[1][self.t_column].values[0] - \
                 self.recent_frames[0][self.t_column].values[0]

        if len(self.pos_columns) != 2:
            raise ValueError('Implemented for 2 dimensions only')
        if self.flow_axis not in self.pos_columns:
            raise ValueError('pos_columns (%r) does not include the specified flow_axis (%s)!' % \
                             (self.pos_columns, self.flow_axis))
        poscols = self.pos_columns[:]
        flow_axis_position = poscols.index(self.flow_axis)
        poscols.remove(self.flow_axis)
        span_axis = poscols[0]

        # Compute velocity field
        disps = pd.DataFrame(dict(span=self.recent_frames[1][span_axis],
            flow=self.recent_frames[1][self.flow_axis] - \
                self.recent_frames[0][self.flow_axis])).dropna()
        # Bin centers
        disps['bin'] = disps.span - disps.span % self.bin_size + self.bin_size / 2.
        grp = disps.groupby('bin')
        # Only use bins that have enough samples
        profcount = grp.flow.count()
        prof = grp.flow.mean()[profcount >= self.minsamples] / dt
        #prof_pos = pandas.DataFrame({self.flow_axis: 0,
        #                             span_axis: pandas.Series(prof.index, index=prof.index)})

        outers = prof.values[0], prof.values[-1]
        prof_ind, prof_vals = list(prof.index), list(prof)
        prof_ind.insert(0, -np.inf)
        prof_vals.insert(0, outers[0])
        prof_ind.append(np.inf)
        prof_vals.append(outers[1])
        prof_ends = pd.Series(prof_vals, index=prof_ind)
        prof_vels = pd.DataFrame({self.flow_axis: prof_ends, span_axis: 0})
        if len(prof) > 0:
            prof_interp = interp1d(prof_vels.index.values, prof_vels[self.pos_columns].values,
                                   'nearest', axis=0)
            if flow_axis_position == 0:
                self.interpolator = lambda x: prof_interp(x[1])
            else:
                self.interpolator = lambda x: prof_interp(x[0])
        else:
            # Not enough samples in any bin
            raise UserWarning(
                'Could not generate velocity field for prediction: not enough tracks or bin_size too small')
            nullvel = np.zeros((len(self.pos_columns),))
            def null_interpolator(x):
                return nullvel
            self.interpolator = null_interpolator
    def predict(self, t1, particle):
        return particle.pos +  \
                     self.interpolator(particle.pos) * (t1 - particle.t)
        #print t1, particle, prediction
        #return prediction
