import os
import itertools
from abc import ABCMeta, abstractmethod, abstractproperty

import pandas as pd
import numpy as np


class FramewiseData(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def put(self, df):
        pass

    @abstractmethod
    def get(self, frame_no):
        pass

    @abstractmethod
    def dump(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractproperty
    def t_column(self):
        pass

    def _validate(self, df):
        if self.t_column not in df.columns:
            raise ValueError("Cannot write frame without a column "
                             "called {0}".format(self.t_column))
        if df[self.t_column].nunique() != 1:
            raise ValueError("Found multiple values for 'frame'. "
                             "Write one frame at a time.")


KEY_PREFIX = 'Frame_'
len_key_prefix = len(KEY_PREFIX)


def code_key(frame_no):
    "Turn the frame_no into a 'natural name' string idiomatic of HDFStore"
    key = '{0}{1}'.format(KEY_PREFIX, frame_no)
    return key


def decode_key(key):
    frame_no = int(key[len_key_prefix + 1:])
    return frame_no


class PandasHDFStore(FramewiseData):
    "Save each frame's data to a node in a pandas HDFStore."

    def __init__(self, filename, t_column='frame'):
        self.filename = os.path.abspath(filename)
        self._t_column = t_column
        self.store = pd.HDFStore(self.filename)

    @property
    def t_column(self):
        return self._t_column

    def put(self, df):
        frame_no = df[self.t_column].iat[0]  # validated to be all the same
        key = code_key(frame_no)
        self.store.put(key, df, data_columns=True)

    def get(self, frame_no):
        key = code_key(frame_no)
        frame = self.store.get(key)
        return frame

    def dump(self):
        keys = self.store.keys()
        keys = sorted(keys, key=decode_key)  # sort numerically
        all_frames = [self.store.get(key) for key in keys]
        return pd.concat(all_frames)

    def __iter__(self):
        return self._build_generator()

    def __del__(self):
        self.store.close()

    def _build_generator(self):
        keys = self.store.keys()
        keys = sorted(keys, key=decode_key)  # sort numerically
        for key in keys:
            frame = self.store.get(key)
            yield frame


class PandasHDFStoreSingleNode(FramewiseData):
    """Save all frames into one large node.

    This implementation is more complex than PandasHDFStore,
    but it simplifies (speeds up?) cross-frame queries,
    like queries for a single probe's entire trajectory."""

    def __init__(self, filename, key, t_column='frame',
                 use_tabular_copy=False):
        self.filename = os.path.abspath(filename)
        self.key = key
        self._t_column = t_column
        self.store = pd.HDFStore(self.filename)

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

    def __iter__(self):
        return self._build_generator()

    def __del__(self):
        self.store.close()

    def _build_generator(self):
        for frame_no in self._inspect_frames():
            frame = self.store.select(self.key, 'frame == %d' % frame_no)
            yield frame

    def _inspect_frames(self):
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
