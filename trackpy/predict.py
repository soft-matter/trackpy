# Copyright 2014, Nathan C. Keim
# keimnathan@gmail.com

"""Tools to improve tracking performance by guessing where a particle will appear next."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
from six.moves import zip

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
    """Predict that particles will not move.

    (Equivalent to standard behavior of linker.)
    """
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

    def link_df(self, *args, **kw):
        """Wrapper for linking.link_df_iter() that causes it to use this predictor.

        As with linking.link_df(), the features data is a single DataFrame.

        Note that this does not wrap linking.link_df(), and does not accept the same
        options as that function. However in most cases it is functionally equivalent.
        """
        args = list(args)
        features = args.pop(0)
        if kw.get('t_column') is None:
            kw['t_column'] = 'frame'
        features_iter = (frame for fnum, frame in features.groupby(kw['t_column']))
        return pd.concat(self.link_df_iter(*([features_iter, ] + args), **kw))

    def observe(self, frame):
        """Examine the latest output of the linker, to update our predictions."""
        pass

    def state(self):
        """Return a representation of the predictor's internal state.

        For diagnostic purposes.
        """
        return None

    def predict(self, t1, particles):
        """Predict the positions of 'particles' at time 't1'"""
        return map(lambda p: p.pos, particles)


class _RecentVelocityPredict(NullPredict):
    def __init__(self, span=1):
        """Use the 'span'+1 most recent frames to make a velocity field."""
        self.recent_frames = deque([], span + 1)

    def state(self):
        return list(self.recent_frames)

    def _compute_velocities(self, frame):
        """Compute velocity field based on a newly-tracked frame."""
        pframe = frame.set_index('particle')
        self.recent_frames.append(pframe)
        if len(self.recent_frames) == 1:
            # Double the first frame. Velocity field will be zero.
            self.recent_frames.append(pframe)
            dt = 1. # Avoid dividing by zero
        else: # Not the first frame
            dt = float(self.recent_frames[-1][self.t_column].values[0] -
                 self.recent_frames[0][self.t_column].values[0])

        # Compute velocity field
        disps = self.recent_frames[-1][self.pos_columns].join(
            self.recent_frames[-1][self.pos_columns] -
                self.recent_frames[0][self.pos_columns], rsuffix='_disp_').dropna()
        positions = disps[self.pos_columns]
        vels = disps[[cn + '_disp_' for cn in self.pos_columns]] / dt
        # 'vels' will have same column names as 'positions'
        vels = vels.rename(columns=lambda n: n[:-6])
        return dt, positions, vels


class NearestVelocityPredict(_RecentVelocityPredict):
    """Predict a particle's position based on the most recent nearby velocity.

    Parameters
    ----------
    initial_guess_positions : Nxd array, optional
    initial_guess_vels : Nxd array, optional
        If specified, these initialize the velocity field with velocity
        samples at the given points.
    span : integer, default 1
        Compute velocity field from the most recent span+1 frames.
    """

    def __init__(self, initial_guess_positions=None,
                 initial_guess_vels=None, span=1):
        super(NearestVelocityPredict, self).__init__(span=span)
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

    def state(self):
        return {'recent_frames': list(self.recent_frames),
                'interpolator': self.interpolator,
                'using_initial_guess': self.use_initial_guess,
                }

    def predict(self, t1, particles):
        poslist, tlist = zip(*[(p.pos, p.t) for p in particles])
        positions = np.array(poslist)
        times = np.array(tlist)
        return (positions + self.interpolator(positions) *
               np.tile(t1 - times, (positions.shape[1], 1)).T)


class DriftPredict(_RecentVelocityPredict):
    """Predict a particle's position based on the mean velocity of all particles.

    Parameters
    ----------
    initial_guess : Array of length d. Otherwise assumed to be zero velocity.
    span : integer, default 1
        Compute velocity field from the most recent span+1 frames.
    """
    def __init__(self, initial_guess=None, span=1):
        super(DriftPredict, self).__init__(span=span)
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
        return (positions + self.vel *
                np.tile(t1 - times, (positions.shape[1], 1)).T)


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
    span : integer, default 1
        Compute velocity field from the most recent span+1 frames.

    Notes
    -----
    - This currently only works for 2D data.
    - Where there were not enough data to make an average velocity (N < minsamples),
        we borrow from the nearest valid bin.
    """
    def __init__(self, bin_size, flow_axis='x', minsamples=20,
                 initial_profile_guess=None, span=1):
        super(ChannelPredict, self).__init__(span=span)
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
            raise ValueError('pos_columns (%r) does not include the specified flow_axis (%s)!' %
                             (self.pos_columns, self.flow_axis))
        poscols = self.pos_columns[:]
        flow_axis_position = poscols.index(self.flow_axis)
        poscols.remove(self.flow_axis)
        span_axis = poscols[0]

        # Make velocity profile
        dt, positions, vels = self._compute_velocities(frame)

        if self.initial_profile_guess is not None:
            ipg = np.asarray(self.initial_profile_guess)
            prof = pd.Series(ipg[:, 1], index=ipg[:, 0])
            self.initial_profile_guess = None  # Don't reuse
        else:
            # Bin centers
            vels['bin'] = (positions[span_axis] - positions[span_axis]
                                                 % self.bin_size + self.bin_size / 2.)
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
                self.interpolator = lambda x: prof_interp(x[:, 1])
            else:
                self.interpolator = lambda x: prof_interp(x[:, 0])
        else:
            # Not enough samples in any bin
            warn('Could not generate velocity field for prediction: '
                 'not enough tracks or bin_size too small')
            nullvel = np.zeros((len(self.pos_columns),))

            def null_interpolator(x):
                return nullvel

            self.interpolator = null_interpolator

    def state(self):
        return {'recent_frames': list(self.recent_frames),
                'interpolator': self.interpolator,
                'initial_profile_guess': self.initial_profile_guess,
                }

    def predict(self, t1, particles):
        poslist, tlist = zip(*[(p.pos, p.t) for p in particles])
        positions = np.array(poslist)
        times = np.array(tlist)
        return (positions + self.interpolator(positions) *
               np.tile(t1 - times, (positions.shape[1], 1)).T)


def instrumented(limit=None):
    """Decorate a predictor class and allow it to record inputs and outputs.

    Use when diagnosing prediction.

    limit : maximum number of recent frames to retain. If None, keep all.

    Examples
    --------

    >>> pred = instrumented()(ChannelPredict)(50, flow_axis='y')
    >>> pred.link_df_iter(...)
    >>> diagnostics = pred.dump()
    """
    def instrumentor(cls):
        class InstrumentedPredictor(cls):
            def __init__(self, *args, **kw):
                super(InstrumentedPredictor, self).__init__(*args, **kw)
                self.diag_observations = deque([], maxlen=limit)
                self.diag_predictions = deque([], maxlen=limit)

            def observe(self, frame):
                self.diag_observations.append(frame)
                return super(InstrumentedPredictor, self).observe(frame)

            def predict(self, t1, particles):
                poslist, tlist, tracklist = zip(*[
                    (p.pos, p.t, p.track.id) for p in particles])
                pdf = pd.DataFrame(np.array(poslist), columns=self.pos_columns)
                pdf[self.t_column] = tlist
                pdf['particle'] = np.array(tracklist, dtype=int)

                prediction = super(InstrumentedPredictor, self).predict(t1, particles)
                pred_df = pd.DataFrame(prediction, columns=self.pos_columns)
                dd = {'t1': t1,
                      'particledf': pdf.join(pred_df, rsuffix='_pred'),
                      'state': self.state()}
                self.diag_predictions.append(dd)

                return prediction

            def dump(self):
                """Report predicted and actual positions.

                Returns list of dictionaries, each containing items
                    "t1": Frame prediction was made *for*
                    "state": Internal state of the predictor, if any
                    "particledf": DataFrame containing positions and
                        predicted positions.
                """
                results = []
                # Latest observation corresponds to the outcome of the
                # most recent linking operation, which corresponds to the
                # most recent element of self.diag_predictions.
                # There may be an extra observation from the beginning
                # of the tracking process, which zip() will ignore.
                for obs, pred in zip(
                        reversed(self.diag_observations),
                        reversed(self.diag_predictions)):
                    dd = pred.copy()
                    dd['particledf'] = dd['particledf'].join(
                        obs.set_index('particle')[self.pos_columns],
                        on='particle', rsuffix='_act')
                    results.append(dd)
                results.reverse()  # Back to chronological order
                return results

        return InstrumentedPredictor
    return instrumentor
