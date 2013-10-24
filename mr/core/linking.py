import itertools
import warnings
import numpy as np
import pandas as pd
import customized_trackpy.tracking as trackpy

class PointNDWithID(trackpy.PointND):
    "Extends pt.PointND to carry meta information from feature identification."
    def __init__(self, t, pos, id):
        trackpy.PointND.__init__(self, t, pos)  # initialize base class
        self.id = id  # unique ID derived from sequential index

class DummyTrack(object):

    track_id = itertools.count(0)

    def __init__(self, point):
       self.id = next(DummyTrack.track_id)
       self.indx = self.id  # redundant, but like trackpy
       if point is not None:
           self.add_point(point)

    def add_point(self, point):
        point.add_to_track(self)
        return self.id

    @classmethod
    def reset_counter(cls, c=0):
        cls.track_id = itertools.count(c)

def link(features, search_range, memory=0, pos_columns=['x', 'y']):
    """Link features into trajectories, assinging a label to each trajectory.

    Notes
    -----
    This function wraps trackpy, Thomas Caswell's implementation
    of the Crocker-Grier algorithm.

    Parameters
    ----------
    frames : iterable that returns a DataFrame for each frame
    search_range : maximum displacement between frames
    memory : largest gap before a trajectory is considered ended
    pos_columns : DataFrame column names (unlimited dimensions)

    Returns
    -------
    features DataFrame, now with additional column of trajectory labels
    """
    MARGIN = 1 # avoid OutOfHashException
    hash_size = features[pos_columns].max() + MARGIN
    features.reset_index(inplace=True, drop=True)
    numbered_frames = (frame for frame in features.groupby('frame'))
    label_generator = \
        link_iterator(numbered_frames, search_range, hash_size, memory)
    features['probe'] = np.nan # placeholder
    while True:
        try:
            frame_no, labels = next(label_generator)
            features['probe'].update(labels)
        except StopIteration:
            break
    return features.sort(['probe', 'frame']).reset_index(drop=True)

def link_iterator(numbered_frames, search_range, hash_size, memory=0,
                  box_size=None, pos_columns=['x', 'y']):
    """Link features into trajectories, assinging a label to each trajectory.

    Notes
    -----
    This function wraps trackpy, Thomas Caswell's implementation
    of the Crocker-Grier algorithm.

    Parameters
    ----------
    frames : iterable that returns 
        a number (frame no.) and an array of positions
    search_range : maximum displacement between frames
    memory : Number of frames through which a probe is allowed to "disappear"
        and reappear and be considered the same probe. Default 0.
    hash_size : Dimensions of the search space, inferred from data by default.
    box_size : A parameter of the underlying algorith, defaults to same
        as search_range, which gives reasonably optimal performance.
    position_cols : DataFrame column names (unlimited dimensions)

    Returns
    -------
    features DataFrame, now with additional column of trajectory labels

    Examples
    --------
    """
    if box_size is None:
        box_size = search_range

    hash_generator = lambda: trackpy.Hash_table(hash_size, box_size)
    levels = _level_generator(numbered_frames, pos_columns)
    labeled_levels = \
        trackpy.link_generator(levels, search_range, hash_generator, memory,
                               track_cls=DummyTrack)
    for level in labeled_levels:
        index = map(lambda x: x.id, level)
        labels = pd.Series(map(lambda x: x.track.id, level), index)
        frame_no = next(iter(level)).t
        _verify_integrity(frame_no, labels) # may issue warnings
        yield frame_no, labels

def _level_generator(numbered_frames, pos_columns):
    for frame_no, frame in numbered_frames:
        build_pt = lambda x: PointNDWithID(frame_no, x[1].values, x[0])
        level = map(build_pt, frame[pos_columns].iterrows())
        yield level

def _verify_integrity(frame_no, labels):
    if labels.duplicated().sum() > 0:
        warnings.warn(
            """There are two probes with the same label in Frame %d.
Proceed with caution.""" % frame_no,
            UserWarning)
    if np.any(labels < 0):
        warnings.warn(
            """Some probes were not labeled in Frame %d. 
Missed probes are labeled -1. Proceed with caution.""" % frame_no,
            UserWarning)
track = link # legacy
