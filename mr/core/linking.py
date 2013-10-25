import itertools
import os
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

def link(features, search_range, memory=0, hash_size=None, box_size=None,
         pos_columns=['x', 'y'], t_column='frame'):
    """Link features into trajectories, assinging a label to each trajectory.

    Notes
    -----
    This function wraps trackpy, Thomas Caswell's implementation
    of the Crocker-Grier algorithm.

    Parameters
    ----------
    frames : a DataFrame of positions to be linked through time
    search_range : maximum displacement between frames
    memory : largest gap before a trajectory is considered ended
    hash_size: region filled by data, detected automatically if not specified
    box_size : parameter to optimize algorithm, set to same as search_range
        if not specified, which gives reasonably optimal performance 
    pos_columns : DataFrame column names (unlimited dimensions)
        Default is ['x', 'y']
    t_column : DataFrame column name. Default is 'frame'.

    Returns
    -------
    features DataFrame, now with additional column of trajectory labels
    """
    if box_size is None:
        box_size = search_range
    if hash_size is None:
        MARGIN = 1 # avoid OutOfHashException
        hash_size = features[pos_columns].max() + MARGIN
    features.reset_index(inplace=True, drop=True)
    numbered_frames = (frame for frame in features.groupby(t_column))
    label_generator = \
        link_iterator(numbered_frames, search_range, hash_size, memory,
                      box_size, pos_columns, t_column)
    features['probe'] = np.nan # placeholder
    while True:
        try:
            frame_no, labels = next(label_generator)
            features['probe'].update(labels)
        except StopIteration:
            break
    return features.sort(['probe', t_column]).reset_index(drop=True)

def link_iterator(numbered_frames, search_range, hash_size, memory=0,
                  box_size=None, pos_columns=['x', 'y'], t_column='frame'):
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
    pos_columns: DataFrame column names (unlimited dimensions)
    t_column : DataFrame column name. Default is 'frame'.

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

CHUNK_SIZE=1000

class LinkOnDisk(object):
    """This helper class manages the process of linking trajectories
    by loading frames one at a time and saving the results one frame
    at a time. This allows infinitely long videos to be analyzed
    using only a modest amount of memory.
    """

    def __init__(self, filename, key, pos_columns=['x', 'y'], 
                 t_column='frame', max_frame=None):
        self.filename = filename
        self.key = key
        self.pos_columns = pos_columns
        self.t_column = t_column
        self.check_file_exists()
        if max_frame is None:
            self.last_frame = self.detect_last_frame()
            print "Detected last frame is Frame %d" % self.last_frame 
        else:
            self.last_frame = max_frame
        # We are assuming that the first frame is Frame 0. If we're wrong,
        # we only waste a little time in the loop.
        self.first_frame = 0
        self.size = self.detect_size() # for hash

    def check_file_exists(self):
        if not os.path.isfile(self.filename):
            raise ValueError(
                "%s is not a path to an HDFStore file." % self.filename)

    def link(self, search_range, memory=0, hash_size=None, box_size=None):
        if hash_size is None:
            MARGIN = 1  # to avoid OutOfHashException
            hash_size = self.size + 1
        numbered_frames = self.numbered_frames()
        label_generator = \
            link_iterator(numbered_frames, search_range, hash_size,
                          memory, box_size, self.pos_columns)
        self.label_generator = label_generator

    def save(self, out_filename, out_key):
        with pd.get_store(out_filename) as out_store:
            with pd.get_store(self.filename) as store:
                while True:
                    try:
                        frame_no, labels = next(self.label_generator)
                    except StopIteration:
                        break
                    # Fetch data (redundantly) this time taking all columns.
                    frame = store.select(self.key, 'frame == %d' % frame_no)
                    frame['probe'] = -1  # an integer placeholder
                    frame['probe'].update(labels)
                    out_store.append(out_key, frame)
                    print "Frame %d written." % frame_no

    def numbered_frames(self):
        with pd.get_store(self.filename) as store:
            for frame_no in xrange(self.first_frame, 1 + self.last_frame):
                frame = store.select(self.key, 'frame == %d' % frame_no, 
                                     columns=self.pos_columns)
                yield frame_no, frame

    def detect_last_frame(self):
        with pd.get_store(self.filename) as store:
            last_frame = 0
            for chunk in store.select_column(self.key, self.t_column, 
                                             iterator=True):
                chunk_max = chunk.max()
                if chunk_max > last_frame:
                    last_frame = chunk_max
        return last_frame

    def detect_size(self):
        with pd.get_store(self.filename) as store:
            size = np.zeros(len(self.pos_columns))
            for chunk in store.select(self.key, iterator=True,
                                      columns=self.pos_columns):
                size = np.maximum(size, chunk.max())
        return size.values 
            

def numbered_frames_from_sql(table, conn, sql_flavor,
                             pos_columns=['x', 'y']):
    pos_columns = ', '.join(pos_columns)
    frame_nos = pd.read_sql("SELECT DISTINCT FRAME FROM %s" % table, conn) 
    frame_nos = frame_nos.values.astype('int')
    for frame_no in frame_nos:
        frame = pd.read_sql("SELECT %s FROM %s WHERE FRAME = %d" %
                              (pos_columns, table, frame_no), conn) 
        yield frame_no, frame


def labeled_frame_to_sql(frame_no_and_labels, table, conn, sql_flavor, t_column):
    frame_no, labels = frame_no_and_labels
    frame[t_column] = frame_no
    frame.to_sql(table, conn, sql_flavor, if_exists='append')

track = link # legacy
