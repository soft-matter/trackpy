# Copyright 2014, Nathan C. Keim
# keimnathan@gmail.com

"""Tools to improve tracking performance by guessing where a particle will appear next."""

from collections import deque
import numpy as np
from scipy.interpolate import NearestNDInterpolator
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

    (If the particle was present in the last 2 frames, its own velocity is used.)
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
            def null_interpolator(*x):
                return np.zeros((len(x),))
            self.interpolator = null_interpolator
    def predict(self, t1, particle):
        prediction = particle.pos +  \
                     self.interpolator(*particle.pos) * (t1 - particle.t)
        #print t1, particle, prediction
        return prediction

