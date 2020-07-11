import itertools

import numpy as np
import pandas as pd


class SubnetOversizeException(Exception):
    '''An :py:exc:`Exception` to be raised when the sub-nets are too big
    to be efficiently linked.  If you get this then either reduce your search range
    or increase :py:attr:`Linker.MAX_SUB_NET_SIZE`'''
    pass


class UnknownLinkingError(Exception):
    pass


def points_to_arr(level):
    """ Convert a list of Points to an ndarray of coordinates """
    return np.array([p.pos for p in level])


def points_from_arr(coords, frame_no, extra_data=None):
    """ Convert an ndarray of coordinates to a list of PointFindLink """
    if extra_data is None:
        return [Point(frame_no, pos) for pos in coords]
    else:
        return [Point(frame_no, pos, extra_data={key: extra_data[key][i]
                                                 for key in extra_data})
                for i, pos in enumerate(coords)]


def coords_from_df(df, pos_columns, t_column):
    """A generator that returns ndarrays of coords from a DataFrame. Assumes
    t_column to be of integer type. Float-typed integers are also accepted.

    Empty frames will be returned as empty arrays of shape (0, ndim)."""

    # This implementation is much faster than using DataFrame.groupby.

    ndim = len(pos_columns)
    times = df[t_column].values
    pos = df[pos_columns].values

    idxs = np.argsort(times, kind="mergesort")  # i.e. stable
    times = times[idxs]
    pos = pos[idxs]

    unique_times, time_counts = np.unique(times, return_counts=True)
    pos_by_frame = np.split(pos, np.cumsum(time_counts)[:-1])

    idx = 0
    for time in range(unique_times[0], unique_times[-1] + 1):
        if time == unique_times[idx]:
            yield time, pos_by_frame[idx]
            idx += 1
        else:
            yield time, np.empty((0, ndim))


def coords_from_df_iter(df_iter, pos_columns, t_column):
    """A generator that returns ndarrays of coords from a generator of
    DataFrames. Also returns the first value of the t_column."""
    ndim = len(pos_columns)

    for df in df_iter:
        if len(df) == 0:
            yield None, np.empty((0, ndim))
        else:
            yield df[t_column].iloc[0], df[pos_columns].values


def verify_integrity(df):
    """Verifies that particle labels are unique for each frame, and that every
    particle is labeled."""
    is_labeled = df['particle'] >= 0
    if not np.all(is_labeled):
        frames = df.loc[~is_labeled, 'frame'].unique()
        raise UnknownLinkingError("Some particles were not labeled "
                                  "in frames {}.".format(frames))
    grouped = df.groupby('frame')['particle']
    try:
        not_equal = grouped.nunique() != grouped.count()
    except AttributeError:  # for older pandas versions
        not_equal = grouped.apply(lambda x: len(pd.unique(x))) != grouped.count()
    if np.any(not_equal):
        where_not_equal = not_equal.index[not_equal].values
        raise UnknownLinkingError("There are multiple particles with the same "
                                  "label in Frames {}.".format(where_not_equal))


class Point:
    '''
    Base class for point (features) used in tracking.  This class
    contains all of the general stuff for interacting with
    :py:class:`~trackpy.linking.Track` objects.


    .. note:: To be used for tracking this class must be sub-classed to provide
    a :py:meth:`distance` function.  Child classes **MUST** call
    :py:meth:`Point.__init__`.  (See :py:class:`~trackpy.linking.PointND` for
    example. )
    '''
    __slots__ = ['_track', 'uuid', 't', 'pos', 'id', 'extra_data', 
                 'forward_cands', 'subnet', 'relocate_neighbors', '__dict__']
    @classmethod
    def reset_counter(cls, c=0):
        cls.counter = itertools.count(c)

    def __init__(self, t, pos, id=None, extra_data=None):
        self._track = None
        self.uuid = next(self.counter)         # unique id for __hash__
        self.t = t
        self.pos = np.asarray(pos)
        self.id = id
        if extra_data is None:
            self.extra_data = dict()
        else:
            self.extra_data = extra_data
        # self.back_cands = []
        self.forward_cands = []
        self.subnet = None
        self.relocate_neighbors = []

    # def __eq__(self, other):
    #     return self.uuid == other.uuid

    # def __neq__(self, other):
    #     return not self.__eq__(other)

    def add_to_track(self, track):
        '''
        :param track: the track to assign to this :py:class:`Point`

        Sets the track of a :py:class:`Point` object.  Raises
        :py:exc:`Exception` if the object is already assigned a track.



        '''
        if self._track is not None:
            raise Exception("trying to add a particle already in a track")
        self._track = track

    def remove_from_track(self, track):
        '''
        :param track: the track to disassociate from this :py:class:`Point`

        Removes this point from the given track. Raises :py:exc:`Exception` if
        particle not associated with the given track.


        '''
        if self._track != track:
            raise Exception("Point not associated with given track")
        track.remove_point(self)

    def in_track(self):
        '''
        :rtype: bool

        Returns if a point is associated with a track '''
        return self._track is not None

    @property
    def track(self):
        """Returns the track that this :class:`Point` is in.  May be `None` """
        return self._track


class TrackUnstored:
    """
    Base class for objects to represent linked tracks.

    Includes logic for adding features to the track, but does
    not store the track's particles in memory.

    Parameters
    ----------
    point : Point or None, optional
        The first feature in the track

    """
    __slots__ = ['id', 'indx', '__dict__']
    @classmethod
    def reset_counter(cls, c=0):
        cls.counter = itertools.count(c)

    def __init__(self, point=None):
        self.id = next(self.counter)
        self.indx = self.id  # redundant, but like trackpy
        if point is not None:
            self.add_point(point)

    def add_point(self, point):
        point.add_to_track(self)

    def incr_memory(self):
        """Mark this track as being remembered for one more frame.

        For diagnostic purposes."""
        try:
            self._remembered += 1
        except AttributeError:
            self._remembered = 1

    def report_memory(self):
        """Report and reset the memory counter (when a link is made).

        For diagnostic purposes."""
        try:
            m = self._remembered
            del self._remembered
            return m
        except AttributeError:
            return 0

    def __repr__(self):
        return "<%s %d>" % (self.__class__.__name__, self.indx)
