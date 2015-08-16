from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import logging
import os
from abc import ABCMeta, abstractmethod, abstractproperty
import warnings

import pandas as pd

logger = logging.getLogger(__name__)


class FramewiseData(object):
    "Abstract base class defining a data container with framewise access."

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

    def dump(self, N=None):
        """Return data from all, or the first N, frames in a single DataFrame

        Parameters
        ----------
        N : integer
            optional; if None, return all frames

        Returns
        -------
        DataFrame
        """
        if N is None:
            return pd.concat(iter(self))
        else:
            i = iter(self)
            return pd.concat((next(i) for _ in range(N)))

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

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

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
    """An interface to an HDF5 file with framewise access, using pandas.

    Save each frame's data to a node in a pandas HDFStore.

    Any additional keyword arguments to the constructor are passed to
    pandas.HDFStore().
    """

    def __init__(self, filename, mode='a', t_column='frame', **kwargs):
        self.filename = os.path.abspath(filename)
        self._t_column = t_column
        self.store = pd.HDFStore(self.filename, mode, **kwargs)

    @property
    def t_column(self):
        return self._t_column

    @property
    def max_frame(self):
        return max(self.frames)

    def put(self, df):
        if len(df) == 0:
            warnings.warn('An empty DataFrame was passed to put(). Continuing.')
            return
        frame_no = df[self.t_column].values[0]  # validated to be all the same
        key = code_key(frame_no)
        # Store data as tabular instead of fixed-format.
        # Make sure remove any prexisting data, so don't really 'append'.
        try:
            self.store.remove(key)
        except KeyError:
            pass
        self.store.put(key, df, format='table')

    def get(self, frame_no):
        key = code_key(frame_no)
        frame = self.store.get(key)
        return frame

    @property
    def frames(self):
        """Returns sorted list of integer frame numbers in file"""
        return self._get_frame_nos()

    def _get_frame_nos(self):
        """Returns sorted list of integer frame numbers in file"""
        # Pandas' store.keys() scans the entire file looking for stored Pandas
        # structures. This is very slow for large numbers of frames.
        # Instead, scan the root level of the file for nodes with names
        # matching our scheme; we know they are DataFrames.
        r = [decode_key(key) for key in self.store.root._v_children.keys() if
             key.startswith(KEY_PREFIX)]
        r.sort()
        return r

    def close(self):
        self.store.close()


class PandasHDFStoreBig(PandasHDFStore):
    """Like PandasHDFStore, but keeps a cache of frame numbers.

    This can give a large performance boost when a file contains thousands
    of frames.

    If a file was made in PandasHDFStore, opening it with this class
    and then closing it will add a cache (if mode != 'r').

    Any additional keyword arguments to the constructor are passed to
    pandas.HDFStore().
    """

    def __init__(self, filename, mode='a', t_column='frame', **kwargs):
        self._CACHE_NAME = '_Frames_Cache'
        self._frames_cache = None
        self._cache_dirty = False  # Whether _frames_cache needs to be written out
        super(PandasHDFStoreBig, self).__init__(filename, mode, t_column,
                                                **kwargs)

    @property
    def frames(self):
        # Hit memory cache, then disk cache
        if self._frames_cache is not None:
            return self._frames_cache
        else:
            try:
                self._frames_cache = list(self.store[self._CACHE_NAME].index.values)
                self._cache_dirty = False
            except KeyError:
                self._frames_cache = self._get_frame_nos()
                self._cache_dirty = True # In memory, but not in file
            return self._frames_cache

    def put(self, df):
        self._invalidate_cache()
        super(PandasHDFStoreBig, self).put(df)

    def rebuild_cache(self):
        """Delete cache on disk and rebuild it."""
        self._invalidate_cache()
        _ = self.frames # Compute cache
        self._flush_cache()

    def _invalidate_cache(self):
        self._frames_cache = None
        try:
            del self.store[self._CACHE_NAME]
        except KeyError: pass

    def _flush_cache(self):
        """Writes frame cache if dirty and file is writable."""
        if (self._frames_cache is not None and self._cache_dirty
                and self.store.root._v_file._iswritable()):
            self.store[self._CACHE_NAME] = pd.DataFrame({'dummy': 1},
                                                        index=self._frames_cache)
            self._cache_dirty = False

    def close(self):
        """Updates cache, writes if necessary, then closes file."""
        if self.store.root._v_file._iswritable():
            _ = self.frames # Compute cache
            self._flush_cache()
        super(PandasHDFStoreBig, self).close()


class PandasHDFStoreSingleNode(FramewiseData):
    """An interface to an HDF5 file with framewise access,
    using pandas, that is faster for cross-frame queries.

    This implementation is more complex than PandasHDFStore,
    but it simplifies (speeds up?) cross-frame queries,
    like queries for a single probe's entire trajectory.

    Any additional keyword arguments to the constructor are passed to
    pandas.HDFStore().
    """

    def __init__(self, filename, key='FrameData', mode='a', t_column='frame',
                 use_tabular_copy=False, **kwargs):
        self.filename = os.path.abspath(filename)
        self.key = key
        self._t_column = t_column
        self.store = pd.HDFStore(self.filename, mode, **kwargs)

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
        if len(df) == 0:
            warnings.warn('An empty DataFrame was passed to put(). Continuing.')
            return
        self._validate(df)
        self.store.append(self.key, df, data_columns=True)

    def get(self, frame_no):
        frame = self.store.select(self.key, '{0} == {1}'.format(
            self._t_column, frame_no))
        return frame

    def dump(self, N=None):
        """Return data from all, or the first N, frames in a single DataFrame

        Parameters
        ----------
        N : integer
            optional; if None, return all frames

        Returns
        -------
        DataFrame
        """
        if N is None:
            return self.store.select(self.key)
        else:
            Nth_frame = self.frames[N - 1]
            return self.store.select(self.key, '{0} <= {1}'.format(
                self._t_column, Nth_frame))

    def close(self):
        self.store.close()

    def __del__(self):
        if hasattr(self, 'store'):
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
    logger.info("Making a tabular copy of %s at %s", (key, tabular_key))
    store.append(tabular_key, store.get(key), data_columns=True)
    return tabular_key
