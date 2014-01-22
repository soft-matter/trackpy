import itertools
import os
import warnings
import numpy as np
import pandas as pd
import trackpy.tracking
from print_update import print_update


link = trackpy.tracking.link_df


class LinkOnDisk(object):
    """This helper class manages the process of linking trajectories
    by loading frames one at a time and saving the results one frame
    at a time. This allows infinitely long videos to be analyzed
    using only a modest amount of memory.
    """

    def __init__(self, filename, key, pos_columns=['x', 'y'], 
                 t_column='frame', max_frame=None, use_tabular_copy=False):
        self.filename = filename
        self.key = key
        self.pos_columns = pos_columns
        self.t_column = t_column
        self.max_frame = max_frame
        self.check_file_exists()
        if use_tabular_copy:
            self.key = make_tabular_copy(self.filename, self.key)
        self.check_node_is_tabular()
        if self.max_frame is None:
            self.last_frame = self.detect_last_frame()
            print "Detected last frame is Frame %d" % self.last_frame 
        else:
            self.last_frame = self.max_frame
        # We are assuming that the first frame is Frame 0. If we're wrong,
        # we only waste a little time in the loop.
        self.first_frame = 0
        self.size = self.detect_size() # for hash

    def check_file_exists(self):
        if not os.path.isfile(self.filename):
            raise ValueError(
                "%s is not a path to an HDFStore file." % self.filename)

    def check_node_is_tabular(self):
        if not is_tabular(self.filename, self.key):
            raise ValueError("""This node is not tabular. Call with use_tabular_copy=True to proceed.""")


    def link(self, search_range, memory=0, hash_size=None, box_size=None,
             verify_integrity=True):
        if hash_size is None:
            MARGIN = 1  # to avoid OutOfHashException
            hash_size = self.size + 1
        numbered_frames = self.numbered_frames()
        label_generator = \
            link_iterator(numbered_frames, search_range, hash_size,
                          memory, box_size, self.pos_columns, self.t_column,
                          verify_integrity)
        self.label_generator = label_generator

    def numbered_frames(self):
        with pd.get_store(self.filename) as store:
            for frame_no in xrange(self.first_frame, 1 + self.last_frame):
                frame = store.select(self.key, 'frame == %d' % frame_no, 
                                     columns=self.pos_columns)
                yield frame_no, frame

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
                    frame['particle'] = -1  # an integer placeholder
                    frame['particle'].update(labels)
                    out_store.append(out_key, frame, 
                                     data_columns=frame.columns)
                    print "Frame %d written with %d particles tracked." \
                        % (frame_no, len(frame))

    def detect_last_frame(self):
        print "Detecting last frame..."
        with pd.get_store(self.filename) as store:
            last_frame = 0
            for chunk in store.select_column(self.key, self.t_column, 
                                             iterator=True):
                chunk_max = chunk.max()
                if chunk_max > last_frame:
                    last_frame = chunk_max
        return last_frame

    def detect_size(self):
        print "Detecting range of position data..."
        with pd.get_store(self.filename) as store:
            size = np.zeros(len(self.pos_columns))
            for chunk in store.select(self.key, iterator=True,
                                      columns=self.pos_columns):
                size = np.maximum(size, chunk.max())
        return size.values 

def is_tabular(filename, key):
    with pd.get_store(filename) as store:
        pandas_type = getattr(getattr(getattr(store._handle.root, key, None),
                                  '_v_attrs', None), 'pandas_type', None)
        return pandas_type == 'frame_table'

def make_tabular_copy(filename, key):
    tabular_key = key + '/tabular'
    print "Making a tabular copy of %s at %s" % (key, tabular_key)
    with pd.get_store(filename) as store:
        store.append(tabular_key, store.get(key), data_columns=['frame'])
    return tabular_key

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
