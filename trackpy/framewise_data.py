import os
from abc import ABCMeta, abstractmethod, abstractproperty

import pandas as pd


class FramewiseData(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def put(self, df):
        pass

    @abstractmethod
    def get(self, frame_no):
        pass

    @abstractproperty
    def frames(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractproperty
    def t_column(self):
        pass

    def __getitem__(self, frame_no):
        return self.get(frame_no)

    def __len__(self):
        return len(self.frames)

    def dump(self):
        """Return data from all frames in a single DataFrame"""
        return pd.concat(iter(self))

    @property
    def max_frame(self):
        return max(self.frames)

    def _validate(self, df):
        if self.t_column not in df.columns:
            raise ValueError("Cannot write frame without a column "
                             "called {0}".format(self.t_column))
        if df[self.t_column].nunique() != 1:
            raise ValueError("Found multiple values for 'frame'. "
                             "Write one frame at a time.")

    def __iter__(self):
        return self._build_generator()

    def _build_generator(self):
        for frame_no in self.frames:
            yield self.get(frame_no)


KEY_PREFIX = 'Frame_'
len_key_prefix = len(KEY_PREFIX)


def code_key(frame_no):
    "Turn the frame_no into a 'natural name' string idiomatic of HDFStore"
    key = '{0}{1}'.format(KEY_PREFIX, frame_no)
    return key


def decode_key(key):
    frame_no = int(key[len_key_prefix:])
    return frame_no


class PandasHDFStore(FramewiseData):
    "Save each frame's data to a node in a pandas HDFStore."

    def __init__(self, filename, mode='a', t_column='frame'):
        self.filename = os.path.abspath(filename)
        self._t_column = t_column
        self.store = pd.HDFStore(self.filename, mode)

    @property
    def t_column(self):
        return self._t_column

    @property
    def max_frame(self):
        return max(self.frames)

    def put(self, df):
        frame_no = df[self.t_column].iat[0]  # validated to be all the same
        key = code_key(frame_no)
        self.store.put(key, df, data_columns=True)

    def get(self, frame_no):
        key = code_key(frame_no)
        frame = self.store.get(key)
        return frame

    @property
    def frames(self):
        """Returns sorted list of integer frame numbers in file"""
        # Pandas' store.keys() scans the entire file looking for stored Pandas
        # structures. This is very slow for large numbers of frames.
        # Instead, scan the root level of the file for nodes with names
        # matching our scheme; we know they are DataFrames.
        r = [decode_key(key) for key in dir(self.store.root) if \
            key.startswith(KEY_PREFIX)]
        r.sort()
        return r

    def close(self):
        self.store.close()

    def __del__(self):
        self.close()


class PandasHDFStoreSingleNode(FramewiseData):
    """Save all frames into one large node.

    This implementation is more complex than PandasHDFStore,
    but it simplifies (speeds up?) cross-frame queries,
    like queries for a single probe's entire trajectory."""

    def __init__(self, filename, key, mode='a', t_column='frame',
                 use_tabular_copy=False):
        self.filename = os.path.abspath(filename)
        self.key = key
        self._t_column = t_column
        self.store = pd.HDFStore(self.filename, mode)

        with pd.get_store(self.filename) as store:
            try:
                store[self.key]
            except KeyError:
                pass
            else:
                self._validate_node(use_tabular_copy)

    @property
    def t_column(self):
        return self._t_column

    def put(self, df):
        self._validate(df)
        self.store.append(self.key, df, data_columns=True)

    def get(self, frame_no):
        frame = self.store.select(self.key, 'frame == %d' % frame_no)
        return frame

    def dump(self):
        return self.store.get(self.key)

    def close(self):
        self.store.close()

    def __del__(self):
        self.close()

    @property
    def frames(self):
        """Returns sorted list of integer frame numbers in file"""
        # I assume one column can fit in memory, which is not ideal.
        # Chunking does not seem to be implemented for select_column.
        frame_nos = self.store.select_column(self.key, self.t_column).unique()
        frame_nos.sort()
        return frame_nos

    def _validate_node(self, use_tabular_copy):
        # The HDFStore might be non-tabular, which means we cannot select a 
        # subset, and this whole structure will not work.
        # For convenience, this can rewrite the table into a tabular node.
        if use_tabular_copy:
            self.key = _make_tabular_copy(self.filename, self.key)

        pandas_type = getattr(getattr(getattr(
            self.store._handle.root, self.key, None), '_v_attrs', None), 
            'pandas_type', None)
        if not pandas_type == 'frame_table':
            raise ValueError("This node is not tabular. Call with "
                             "use_tabular_copy=True to proceed.")


def _make_tabular_copy(store, key):
    """Copy the contents nontabular node in a pandas HDFStore
    into a tabular node"""
    tabular_key = key + '/tabular'
    print "Making a tabular copy of %s at %s" % (key, tabular_key)
    store.append(tabular_key, store.get(key), data_columns=True)
    return tabular_key
