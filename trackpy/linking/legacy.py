import logging
from warnings import warn
import itertools
import functools

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..try_numba import NUMBA_AVAILABLE
from ..utils import pandas_sort, validate_tuple, is_isotropic
from .utils import (TrackUnstored, points_to_arr, UnknownLinkingError,
                    SubnetOversizeException)
from .subnetlinker import (recursive_linker_obj, nonrecursive_link, drop_link,
                           numba_link)

logger = logging.getLogger(__name__)


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
    @classmethod
    def reset_counter(cls, c=0):
        cls.counter = itertools.count(c)

    def __init__(self):
        self._track = None
        self.uuid = next(self.counter)         # unique id for __hash__

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


class Track(TrackUnstored):
    '''
    Base class for objects to represent linked tracks.

    Includes logic for adding, removing features to the track.  This can
    be sub-classed to provide additional track level computation as
    needed.

    Parameters
    ----------
    point : Point or None, optional
        The first feature in the track

    '''
    def __init__(self, point=None):
        self.points = []
        super().__init__(point)

    def __iter__(self):
        return self.points.__iter__()

    def __len__(self):
        return len(self.points)

    def __eq__(self, other):
        return self.index == other.index

    def __neq__(self, other):
        return not self.__eq__(other)
    __hash__ = None

    def add_point(self, point):
        '''
        :param point: point to add
        :type point:  :py:class:`~trackpy.linking.Point`

        Appends the point to this track. '''
        self.points.append(point)
        point.add_to_track(self)

    def remove_point(self, point):
        '''
        :param point: point to remove from this track
        :type point:  :py:class:`~trackpy.linking.Point`

        removes a point from this track'''
        self.points.remove(point)
        point._track = None

    def last_point(self):
        '''
        :rtype: :py:class:`~trackpy.linking.Point`

        Returns the last point on the track'''
        return self.points[-1]


class PointND(Point):
    '''
    Version of :class:`Point` for tracking in flat space with
    non-periodic boundary conditions.

    Parameters
    ----------
    t : scalar
        a time-like variable.

    pos : array-like
        position of feature

    id : int, optional
        external unique ID
    '''

    def __init__(self, t, pos, id=None):
        Point.__init__(self)                  # initialize base class
        self.t = t                            # time
        self.pos = np.asarray(pos)            # position in ND space
        self.id = id

    def distance(self, other_point):
        '''
        :param other_point: point to get distance to.
        :type other_point: :py:class:`~trackpy.linking.Point`

        Returns the absolute distance between this point and other_point

        '''
        return np.sqrt(np.sum((self.pos - other_point.pos) ** 2))

    def __str__(self):
        return "({t}, {p})".format(t=self.t, p=self.pos)

    def __repr__(self):
        coords = '(' + (', '.join(["{:.3f}"]*len(self.pos))).format(*self.pos) + ')'
        track = " in Track %d" % self.track.indx if self.track else ""
        return "<%s at %d, " % (self.__class__.__name__, self.t) + coords + track + ">"


class PointDiagnostics:
    """Mixin to add memory diagnostics collection to a Point object."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diag = {}

    def add_to_track(self, track):
        super().add_to_track(track)
        # See the note in the memory section of Linker.link(). If this link
        # is from memory, the track knows how many frames were skipped.
        memcount = track.report_memory()
        if memcount > 0:
            self.diag['remembered'] = memcount

class PointNDDiagnostics(PointDiagnostics, PointND):
    """Version of :class:`PointND` that collects diagnostic information
    during tracking.
    """
    pass


class TreeFinder:
    def __init__(self, points, search_range):
        """Takes a list of particles."""
        self.ndim = len(search_range)
        self.search_range = np.atleast_2d(search_range)
        if not isinstance(points, list):
            points = list(points)
        self.points = points
        self.rebuild()

    def __len__(self):
        return len(self.points)

    def add_point(self, pt):
        self.points.append(pt)
        self._clean = False

    @property
    def kdtree(self):
        if not self._clean:
            self.rebuild()
        return self._kdtree

    def rebuild(self, coord_map=None):
        """Rebuilds tree from ``points`` attribute.

        coord_map : function, optional

            Called with a list of N Point instances, returns their
            "effective" locations, as an N x d array (or list of tuples).
            Used for prediction (see "predict" module).

        rebuild() needs to be called after ``add_point()`` and
        before tree is used for spatial queries again (i.e. when
        memory is turned on).
        """
        if coord_map is None:
            coordmap_as_arr = points_to_arr
        else:
            coordmap_as_arr = lambda x: np.asarray(list(coord_map(x)))
        if len(self.points) == 0:
            self._kdtree = None
        else:
            coords = coordmap_as_arr(self.points) / self.search_range
            self._kdtree = cKDTree(coords, 15)
        # This could be tuned
        self._clean = True

    @property
    def coords_rescaled(self):
        if self._clean:
            if self._kdtree is None:
                return np.empty((0, self.ndim))
            else:
                return self._kdtree.data
        else:
            return points_to_arr(self.points) / self.search_range


class HashTable:
    """Basic hash table for fast look up of particles in neighborhood.

    Parameters
    ----------
    dims : ND tuple
        the range of the data to be put in the hash table.
        0<data[k]<dims[k]

    box_size : float
        how big each box should be in data units.
        The same scale is used for all dimensions


    """
    class Out_of_hash_excpt(Exception):
        """
        :py:exc:`Exception` for indicating that a particle is outside of the
        valid range for this hash table."""
        pass

    def __init__(self, dims, box_size):
        '''
        Sets up the hash table

        '''
        # the dimensions of the data
        self.dims = dims
        # the size of boxes to use in the units of the data
        self.box_size = box_size
        self.hash_dims = np.ceil(np.array(dims) / box_size)

        self.hash_table = [[] for j in range(int(np.prod(self.hash_dims)))]
        # how many spatial dimensions
        self.spat_dims = len(dims)
        self.cached_shifts = None
        self.cached_rrange = None
        self.strides = np.cumprod(
                           np.concatenate(([1], self.hash_dims[1:])))[::-1]
        self.points = []

    def get_region(self, point, rrange):
        '''
        Returns all the particles within the region of maximum radius
        rrange in data units.  This may return Points that are farther
        than rrange.

        Parameters
        ----------
        point : Point
            point to find the features around

        rrange: float
            the size of the ball to search in data units.


        '''
        hash_size = self.hash_dims
        center = np.floor(point.pos / self.box_size)
        if any(center >= hash_size) or any(center < 0):
            raise Hash_table.Out_of_hash_excpt("cord out of range")

        rrange = int(np.ceil(rrange / self.box_size))

        # check if we have already computed the shifts
        if rrange == self.cached_rrange and self.cached_shifts is not None:
            shifts = self.cached_shifts   # if we have, use them
        # Other wise, generate them
        else:
            if self.spat_dims == 2:
                shifts = [np.array([j, k])
                          for j in range(-rrange, rrange + 1)
                          for k in range(-rrange, rrange + 1)]
            elif self.spat_dims == 3:
                shifts = [np.array([j, k, m])
                          for j in range(-rrange, rrange + 1)
                          for k in range(-rrange, rrange + 1)
                          for m in range(-rrange, rrange + 1)]
            else:
                raise NotImplementedError('only 2 and 3 dimensions implemented')
            self.cached_rrange = rrange   # and save them
            self.cached_shifts = shifts
        region = []

        for s in shifts:

            cord = center + s
            if any(cord >= hash_size) or any(cord < 0):
                continue
            indx = int(sum(cord * self.strides))
            region.extend(self.hash_table[indx])
        return region

    def add_point(self, point):
        """
        Adds the `point` to the hash table.

        Assumes that :py:attr:`point.pos` exists and is the array-like.

        Parameters
        ----------
        point : Point
            object representing the feature to add to the hash table

        """
        cord = np.floor(np.asarray(point.pos) / self.box_size)
        hash_size = self.hash_dims
        if any(cord >= hash_size) or any(cord < 0):
            raise Hash_table.Out_of_hash_excpt("cord out of range")
        indx = int(sum(cord * self.strides))
        self.hash_table[indx].append(point)
        self.points.append(point)

    def __len__(self):
        return len(self.points)

Hash_table = HashTable


def link(levels, search_range, hash_generator, memory=0, track_cls=None,
         neighbor_strategy='BTree', link_strategy='recursive'):
    """Link features into trajectories, assigning a label to each trajectory.

    This function is deprecated and lacks some recently-added options,
    though it is still accurate. Use link_df or link_iter.

    Parameters
    ----------
    levels : iterable of iterables containing Points objects
        e.g., a list containing lists with the Points in each frame
    search_range : tuple
        the maximum distance features can move between frames, per dimension
    hash_generator : a function that returns a HashTable
        only used if neighbor_strategy is set to 'BTree' (default)
    memory : integer
        the maximum number of frames during which a feature can vanish,
        then reppear nearby, and be considered the same particle. 0 by default.
    neighbor_strategy : {'BTree', 'KDTree'}
        algorithm used to identify nearby features
    link_strategy : {'recursive', 'nonrecursive', 'numba', 'drop', 'auto'}
        algorithm used to resolve subnetworks of nearby particles
        'auto' uses numba if available
        'drop' causes particles in subnetworks to go unlinked

    Returns
    -------
    tracks : list of Track (or track_cls) objects

    See Also
    --------
    link_df, link_iter
    """
    # An informative error to help newbies who go astray
    if isinstance(levels, pd.DataFrame):
        raise TypeError("Instead of link, use link_df, which accepts "
                        "pandas DataFrames.")
    if track_cls is None:
        track_cls = Track  # stores Points
    label_generator = link_iter(iter(levels), search_range, memory=memory,
                                neighbor_strategy=neighbor_strategy,
                                link_strategy=link_strategy,
                                track_cls=track_cls,
                                hash_generator=hash_generator)
    labels = list(label_generator)
    points = [level for level_list in levels for level in level_list]  # flat
    points = pd.Series(points)
    labels = [label.track.indx for label_list in labels
              for label in label_list]  # flat
    grouped = points.groupby(labels)
    representative_points = grouped.first()  # one point from each Track
    tracks = representative_points.apply(lambda x: x.track)
    return tracks


def link_df(features, search_range, memory=0,
            neighbor_strategy='KDTree', link_strategy='auto',
            predictor=None, adaptive_stop=None, adaptive_step=0.95,
            copy_features=False, diagnostics=False, pos_columns=None,
            t_column=None, hash_size=None, box_size=None,
            verify_integrity=True, retain_index=False):
    """Link features into trajectories, assigning a label to each trajectory.

    Parameters
    ----------
    features : DataFrame
        Must include any number of column(s) for position and a column of
        frame numbers. By default, 'x' and 'y' are expected for position,
        and 'frame' is expected for frame number. See below for options to use
        custom column names. After linking, this DataFrame will contain a
        'particle' column.
    search_range : float
        the maximum distance features can move between frames
    memory : integer
        the maximum number of frames during which a feature can vanish,
        then reppear nearby, and be considered the same particle. 0 by default.
    neighbor_strategy : {'KDTree', 'BTree'}
        algorithm used to identify nearby features
    link_strategy : {'recursive', 'nonrecursive', 'numba', 'drop', 'auto'}
        algorithm used to resolve subnetworks of nearby particles
        'auto' uses numba if available
        'drop' causes particles in subnetworks to go unlinked
    predictor : function, optional
        Improve performance by guessing where a particle will be in
        the next frame.

        For examples of how this works, see the "predict" module.
    adaptive_stop : float, optional
        If not None, when encountering an oversize subnet, retry by progressively
        reducing search_range until the subnet is solvable. If search_range
        becomes <= adaptive_stop, give up and raise a SubnetOversizeException.
    adaptive_step : float, optional
        Reduce search_range by multiplying it by this factor.
    copy_features : boolean
        Leave the original features DataFrame intact (slower, uses more memory)
    diagnostics : boolean
        Collect details about how each particle was linked, and return as
        columns in the output DataFrame. Implies copy=True.
    pos_columns : DataFrame column names (unlimited dimensions)
        Default is ['x', 'y']
    t_column : DataFrame column name
        Default is 'frame'
    hash_size : sequence
        For 'BTree' mode only. Define the shape of the search region.
        If None (default), infer shape from range of data.
    box_size : sequence
        For 'BTree' mode only. Define the parition size to optimize
        performance. If None (default), the search_range is used, which is
        a reasonable guess for best performance.
    verify_integrity : boolean
        False by default for fastest performance.
        Use True if you suspect a bug in linking.
    retain_index : boolean
        By default, the index is reset to be sequential. To keep the original
        index, set to True. Default is fine unless you devise a special use.

    Returns
    -------
    trajectories : DataFrame
        This is the input features DataFrame, now with a new column labeling
        each particle with an ID number. This is not a copy; the original
        features DataFrame is modified.
    """
    # Assign defaults. (Do it here to avoid "mutable defaults" issue.)
    if pos_columns is None:
        pos_columns = ['x', 'y']
    if t_column is None:
        t_column = 'frame'
    search_range = validate_tuple(search_range, len(pos_columns))
    if hash_size is None:
        MARGIN = 1  # avoid OutOfHashException
        hash_size = features[pos_columns].max() + MARGIN

    # Group the DataFrame by time steps and make a 'level' out of each
    # one, using the index to keep track of Points.
    if retain_index:
        orig_index = features.index.copy()  # Save it; restore it at the end.
    features.reset_index(inplace=True, drop=True)
    levels = _gen_levels_df(features, pos_columns, t_column, diagnostics)
    labeled_levels = link_iter(
        levels, search_range, memory=memory, predictor=predictor,
        adaptive_stop=adaptive_stop, adaptive_step=adaptive_step,
        neighbor_strategy=neighbor_strategy, link_strategy=link_strategy,
        hash_size=hash_size, box_size=box_size)

    if diagnostics:
        features = strip_diagnostics(features)  # Makes a copy
    elif copy_features:
        features = features.copy()

    # Do the tracking, and update the DataFrame after each iteration.
    features['particle'] = np.nan  # placeholder
    for level in labeled_levels:
        if len(level) == 0:
            continue
        index = [x.id for x in level]
        labels = pd.Series([x.track.id for x in level], index)
        frame_no = next(iter(level)).t  # uses an arbitary element from the set
        if verify_integrity:
            # This checks that the labeling is sane and tries
            # to raise informatively if some unknown bug in linking
            # produces a malformed labeling.
            _verify_integrity(frame_no, labels)
            # an additional check particular to link_df
            if len(labels) > len(features[features[t_column] == frame_no]):
                raise UnknownLinkingError("There are more labels than "
                                          "particles to be labeled in Frame "
                                          "{}.".format(frame_no))
        features['particle'].update(labels)
        if diagnostics:
            _add_diagnostic_columns(features, level)

        logger.info("Frame %d: %d trajectories present", frame_no, len(labels))

    if retain_index:
        features.index = orig_index
        # And don't bother to sort -- user must be doing something special.
    else:
        pandas_sort(features, ['particle', t_column], inplace=True)
        features.reset_index(drop=True, inplace=True)
    return features


def link_df_iter(features, search_range, memory=0,
            neighbor_strategy='KDTree', link_strategy='auto',
            predictor=None, adaptive_stop=None, adaptive_step=0.95,
            diagnostics=False, pos_columns=None,
            t_column=None, hash_size=None, box_size=None,
            verify_integrity=True, retain_index=False):
    """Link features into trajectories, assigning a label to each trajectory.

    Parameters
    ----------
    features : iterable of DataFrames
        Each DataFrame must include any number of column(s) for position and a
        column of frame numbers. By default, 'x' and 'y' are expected for
        position, and 'frame' is expected for frame number. See below for
        options to use custom column names.
    search_range : float
        the maximum distance features can move between frames
    memory : integer
        the maximum number of frames during which a feature can vanish,
        then reppear nearby, and be considered the same particle. 0 by default.
    neighbor_strategy : {'KDTree', 'BTree'}
        algorithm used to identify nearby features. Note that when using
        BTree, you must specify hash_size
    link_strategy : {'recursive', 'nonrecursive', 'numba', 'drop', 'auto'}
        algorithm used to resolve subnetworks of nearby particles
        'auto' uses numba if available
        'drop' causes particles in subnetworks to go unlinked
    predictor : function, optional
        Improve performance by guessing where a particle will be in the
        next frame.

        For examples of how this works, see the "predict" module.
    adaptive_stop : float, optional
        If not None, when encountering an oversize subnet, retry by progressively
        reducing search_range until the subnet is solvable. If search_range
        becomes <= adaptive_stop, give up and raise a SubnetOversizeException.
    adaptive_step : float, optional
        Reduce search_range by multiplying it by this factor.
    diagnostics : boolean
        Collect details about how each particle was linked, and return as
        columns in the output DataFrame.
    pos_columns : DataFrame column names (unlimited dimensions)
        Default is ['x', 'y']
    t_column : DataFrame column name
        Default is 'frame'
    hash_size : sequence
        For 'BTree' mode only. Define the shape of the search region.
    box_size : sequence
        For 'BTree' mode only. Define the parition size to optimize
        performance. If None (default), the search_range is used, which is
        a reasonable guess for best performance.
    verify_integrity : boolean
        False by default, for fastest performance.
        Use True if you suspect a bug in linking.
    retain_index : boolean
        By default, the index is reset to be sequential. To keep the original
        index, set to True. Default is fine unless you devise a special use.

    Returns
    -------
    trajectories : DataFrame
        This is the input features DataFrame, now with a new column labeling
        each particle with an ID number for each frame.
    """
    # Assign defaults. (Do it here to avoid "mutable defaults" issue.)
    if pos_columns is None:
        pos_columns = ['x', 'y']
    if t_column is None:
        t_column = 'frame'
    search_range = validate_tuple(search_range, len(pos_columns))

    # Group the DataFrame by time steps and make a 'level' out of each
    # one, using the index to keep track of Points.

    # Non-destructively check the type of the first item of features
    feature_iter, feature_checktype_iter = itertools.tee(iter(features))
    try:  # If it quacks like a DataFrame...
        next(feature_checktype_iter).reset_index()
    except AttributeError:
        raise ValueError("Features data must be an iterable of DataFrames, one per "
                         "video frame. Use link_df() if you have a single DataFrame "
                         "describing multiple frames.")
    del feature_checktype_iter  # Otherwise pipes will back up.

    # To allow retain_index
    features_for_reset, features_forindex = itertools.tee(feature_iter)
    index_iter = (fr.index.copy() for fr in features_forindex)
    # To allow extra columns to be recovered later
    features_forlinking, features_forpost = itertools.tee(
        frame.reset_index(drop=True) for frame in features_for_reset)
    # make a generator over the frames
    levels = (_build_level(frame, pos_columns, t_column, diagnostics=diagnostics)
                         for frame in features_forlinking)

    # make a generator of the levels post-linking
    labeled_levels = link_iter(
        levels, search_range, memory=memory, predictor=predictor,
        adaptive_stop=adaptive_stop, adaptive_step=adaptive_step,
        neighbor_strategy=neighbor_strategy, link_strategy=link_strategy,
        hash_size=hash_size, box_size=box_size)

    # Re-assemble the features data, now with track labels and (if desired)
    # the original index.
    for labeled_level, source_features, old_index in zip(
            labeled_levels, features_forpost, index_iter):
        features = source_features.copy()
        features['particle'] = np.nan  # placeholder
        index = [x.id for x in labeled_level]
        labels = pd.Series([x.track.id for x in labeled_level], index)
        # uses an arbitary element from the set
        frame_no = next(iter(labeled_level)).t
        if verify_integrity:
            # This checks that the labeling is sane and tries
            # to raise informatively if some unknown bug in linking
            # produces a malformed labeling.
            _verify_integrity(frame_no, labels)
            # additional checks particular to link_df_iter
            if not all(frame_no == source_features[t_column].values):
                raise UnknownLinkingError(("The features passed for Frame {} "
                                          "do not all share the same frame "
                                          "number.").format(frame_no))
            if len(labels) > len(features):
                raise UnknownLinkingError("There are more labels than "
                                          "particles to be labeled in Frame "
                                           "{}.".format(frame_no))
        features['particle'].update(labels)
        if diagnostics:
            _add_diagnostic_columns(features, labeled_level)

        if retain_index:
            features.index = old_index
            # TODO: don't run index.copy() even when retain_index is false
        else:
            pandas_sort(features, 'particle', inplace=True)
            features.reset_index(drop=True, inplace=True)

        logger.info("Frame %d: %d trajectories present", frame_no, len(labels))

        yield features


def _build_level(frame, pos_columns, t_column, diagnostics=False):
    """Return PointND objects for a DataFrame of points.

    Parameters
    ----------
    frame : DataFrame
        Unlinked points data.
    pos_columns : list
        Names of position columns in "frame"
    t_column : string
        Name of time column in "frame"
    diagnostics : boolean, optional
        Whether resulting point objects should collect diagnostic information.
    """
    if diagnostics:
        point_cls = PointNDDiagnostics
    else:
        point_cls = PointND
    # This is the only chance to do this, as only here the point_cls is known.
    if not hasattr(point_cls, 'counter'):
        point_cls.reset_counter()
    return list(map(point_cls, frame[t_column],
                    frame[pos_columns].values, frame.index))


def _gen_levels_df(df, pos_columns, t_column, diagnostics=False):
    """Return a generator of PointND objects for a DataFrame of points.

    The DataFrame is assumed to contain integer framenumbers. For a missing
    frame number, an empty list is returned.

    Parameters
    ----------
    df : DataFrame
        Unlinked points data for all frames.
    pos_columns : list
        Names of position columns in "frame"
    t_column : string
        Name of time column in "frame"
    diagnostics : boolean, optional
        Whether resulting point objects should collect diagnostic information.
    """
    grouped = iter(df.groupby(t_column))
    cur_frame, frame = next(grouped)
    cur_frame += 1.5  # set counter to 1.5 for issues with e.g. 1.000001
    yield _build_level(frame, pos_columns, t_column, diagnostics)

    for frame_no, frame in grouped:
        while cur_frame < frame_no:
            cur_frame += 1
            yield []

        cur_frame += 1
        yield _build_level(frame, pos_columns, t_column, diagnostics)


def _add_diagnostic_columns(features, level):
    """Copy the diagnostic information stored in each particle to the
    corresponding columns in 'features'. Create columns as needed."""
    diag = pd.DataFrame({x.id: x.diag for x in level}, dtype=object).T
    diag.columns = ['diag_' + cn for cn in diag.columns]
    for cn in diag.columns:
        if cn not in features.columns:
            features[cn] = pd.Series(np.nan, dtype=float, index=features.index)
    features.update(diag)


def strip_diagnostics(tracks):
    """Remove diagnostic information from a tracks DataFrame.

    This returns a copy of the DataFrame. Columns with names that start
    with "diag_" are excluded."""
    base_cols = [cn for cn in tracks.columns if not cn.startswith('diag_')]
    return tracks.reindex(columns=base_cols)


def _verify_integrity(frame_no, labels):
    if labels.duplicated().sum() > 0:
        raise UnknownLinkingError(
            "There are two particles with the same label in Frame {}.".format(
                frame_no))
    if np.any(labels < 0):
        raise UnknownLinkingError("Some particles were not labeled "
                                  "in Frame {}.".format(frame_no))


def link_iter(levels, search_range, memory=0,
              neighbor_strategy='KDTree', link_strategy='auto',
              hash_size=None, box_size=None, predictor=None,
              adaptive_stop=None, adaptive_step=0.95,
              track_cls=None, hash_generator=None):
    """Link features into trajectories, assigning a label to each trajectory.

    This function is a generator which yields at each step the Point
    objects for the current level.  These objects know what trajectory
    they are in.

    Parameters
    ----------
    levels : iterable of iterables containing Points objects
        e.g., a list containing lists with the Points in each frame
    search_range : tuple
        the maximum distance features can move between frames, per dimension
    memory : integer
        the maximum number of frames during which a feature can vanish,
        then reppear nearby, and be considered the same particle. 0 by default.
    neighbor_strategy : {'KDTree', 'BTree'}
        algorithm used to identify nearby features
    link_strategy : {'recursive', 'nonrecursive', 'numba', 'drop', 'auto'}
        algorithm used to resolve subnetworks of nearby particles
        'auto' uses numba if available
        'drop' causes particles in subnetworks to go unlinked
    hash_size : sequence
        For 'BTree' mode only. Define the shape of the search region.
        (Higher-level wrappers of link infer this from the data.)
    box_size : sequence
        For 'BTree' mode only. Define the parition size to optimize
        performance. If None (default), the search_range is used, which is
        a reasonable guess for best performance.
    predictor : function, optional
        Improve performance by guessing where a particle will be in the
        next frame.
        For examples of how this works, see the "predict" module.
    adaptive_stop : float, optional
        If not None, when encountering an oversize subnet, retry by progressively
        reducing search_range until the subnet is solvable. If search_range
        becomes <= adaptive_stop, give up and raise a SubnetOversizeException.
    adaptive_step : float, optional
        Reduce search_range by multiplying it by this factor.
    track_cls : class, optional
        for special uses, you can specify a custom class that holds
        each Track
    hash_generator : function, optional
        a function that returns a HashTable, included for legacy support.
        Specifying hash_size and box_size (above) fully defined a HashTable.

    Returns
    -------
    cur_level : iterable of Point objects
        The labeled points at each level.
    """
    if not hasattr(search_range, '__iter__'):
        # We need to know the dimensionality of the image, but this information
        # is not accessible at this point. Guess 2D and give a warning.
        search_range = (search_range,) * 2
        warn("In link and link_iter, search_range should be given as a "
             "tuple with the same length as the number of dimensions in "
             "the image. We now assume the dimensionality to be 2D.")
    linker = Linker(search_range, memory=memory, neighbor_strategy=neighbor_strategy,
                 link_strategy=link_strategy, hash_size=hash_size,
                 box_size=box_size, predictor=predictor,
                 adaptive_stop=adaptive_stop, adaptive_step=adaptive_step,
                 track_cls=track_cls, hash_generator=hash_generator)
    return linker.link(levels)

class Linker:
    """See link_iter() for a description of parameters."""
    # Largest subnet we will attempt to solve.
    MAX_SUB_NET_SIZE = 30
    # For adaptive search, subnet linking should fail much faster.
    MAX_SUB_NET_SIZE_ADAPTIVE = 15

    def __init__(self, search_range, memory=0,
              neighbor_strategy='KDTree', link_strategy='auto',
              hash_size=None, box_size=None, predictor=None,
              adaptive_stop=None, adaptive_step=0.95,
              track_cls=None, hash_generator=None):
        self.search_range = search_range
        self.memory = memory
        self.predictor = predictor
        self.adaptive_stop = adaptive_stop
        self.adaptive_step = adaptive_step
        if track_cls is None:
            track_cls = TrackUnstored  # does not store Points
        track_cls.reset_counter()
        self.track_cls = track_cls
        self.hash_generator = hash_generator
        self.neighbor_strategy = neighbor_strategy

        self.diag = False  # Whether to save diagnostic info

        if self.hash_generator is None:
            if neighbor_strategy == 'BTree':
                if hash_size is None:
                    raise ValueError("In 'BTree' mode, you must specify hash_size")
                if box_size is None:
                    box_size = search_range[0]
            self.hash_generator = lambda: Hash_table(hash_size, box_size)

        linkers = {'recursive': recursive_linker_obj,
                   'nonrecursive': nonrecursive_link,
                   'drop': drop_link}
        if NUMBA_AVAILABLE:
            linkers['numba'] = numba_link
            linkers['auto'] = linkers['numba']
        else:
            linkers['auto'] = linkers['recursive']
        try:
            self.subnet_linker = linkers[link_strategy]
        except KeyError:
            raise ValueError("link_strategy must be one of: " + ', '.join(linkers.keys()))

        if self.neighbor_strategy not in ['KDTree', 'BTree']:
            raise ValueError("neighbor_strategy must be 'KDTree' or 'BTree'")
        if self.neighbor_strategy == 'BTree' and not is_isotropic(self.search_range):
            raise ValueError("The neighbor_strategy 'BTree' cannot deal with "
                             "anisotropic search ranges. Please use 'KDTree'.")

        if self.adaptive_stop is not None:
            # adaptive_stop is a fraction of search range
            self.adaptive_stop = self.adaptive_stop / np.min(self.search_range)
            if 1 * self.adaptive_stop <= 0:
                raise ValueError("adaptive_stop must be positive.")
            self.max_subnet_size = self.MAX_SUB_NET_SIZE_ADAPTIVE
        else:
            self.max_subnet_size = self.MAX_SUB_NET_SIZE

        if 1 * self.adaptive_step <= 0 or 1 * self.adaptive_step >= 1:
            raise ValueError("adaptive_step must be between "
                             "0 and 1 non-inclusive.")

        self.subnet_counter = 0  # Unique ID for each subnet

    def link(self, levels):
        level_iter = iter(levels)
        prev_level = next(level_iter)
        prev_set = set(prev_level)

        # Only save diagnostic info if it's possible. This saves
        # 1-2% execution time and significant memory.
        # We just check the first particle in the first level.
        self.diag = hasattr(next(iter(prev_level)), 'diag')

        # Make a Hash / Tree for the first level.
        if self.neighbor_strategy == 'BTree':
            prev_hash = self.hash_generator()
            for p in prev_set:
                prev_hash.add_point(p)
        elif self.neighbor_strategy == 'KDTree':
            prev_hash = TreeFinder(prev_level, self.search_range)

        for p in prev_set:
            p.forward_cands = []

        try:
            # Start ID numbers from zero, incompatible with multithreading.
            self.track_cls.reset_counter()
        except AttributeError:
            # must be using a custom Track class without this method
            pass

        # Assume everything in first level starts a Track.
        # Iterate over prev_level, not prev_set, because order -> track ID.
        for p in prev_level:
            self.track_cls(p)
        self.mem_set = set()

        # Initialize memory with empty sets.
        mem_history = []
        for j in range(self.memory):
            mem_history.append(set())

        yield list(prev_set)  # Short-circuit the loop on first call.

        for cur_level in levels:
            # Create the set for the destination level.
            cur_set = set(cur_level)
            tmp_set = set(cur_level)  # copy used in next loop iteration

            # First, a bit of unfinished business:
            # If prediction is enabled, we need to update the positions in prev_hash
            # to where we think they'll be in the frame corresponding to cur_level.
            if self.predictor is not None:
                # This only works for KDTree right now, because KDTree can store particle
                # positions in a separate data structure from the PointND instances.
                if not isinstance(prev_hash, TreeFinder):
                    raise NotImplementedError(
                        'Prediction works with the "KDTree" neighbor_strategy only.')
                # Get the time of cur_level from its first particle
                t_next = list(itertools.islice(cur_level, 0, 1))[0].t
                targeted_predictor = functools.partial(self.predictor, t_next)
                prev_hash.rebuild(coord_map=targeted_predictor) # Rewrite positions

            # Now we can process the new particles.
            # Make a Hash / Tree for the destination level.
            if self.neighbor_strategy == 'BTree':
                cur_hash = self.hash_generator()
                for p in cur_set:
                    cur_hash.add_point(p)
            elif self.neighbor_strategy == 'KDTree':
                cur_hash = TreeFinder(cur_level, self.search_range)

            # Set up attributes for keeping track of possible connections.
            for p in cur_set:
                p.back_cands = []
                p.forward_cands = []

            # Sort out what can go to what. This uses the search_range to
            # rescale distances between particles in between 0 and 1. From here
            # on, search_range = 1.
            assign_candidates(cur_hash, prev_hash, self.search_range,
                              self.neighbor_strategy)

            # sort the candidate lists by distance
            for p in cur_set:
                p.back_cands.sort(key=lambda x: x[1])
            for p in prev_set:
                p.forward_cands.sort(key=lambda x: x[1])

            # Note that this modifies cur_set, prev_set, but that's OK.
            spl, dpl = self._assign_links(cur_set, prev_set, search_range=1.)

            new_mem_set = set()
            for sp, dp in zip(spl, dpl):
                # Do linking
                if sp is not None and dp is not None:
                    sp.track.add_point(dp)
                    if sp in self.mem_set:  # Very rare
                        self.mem_set.remove(sp)
                elif sp is None:
                    # if unclaimed destination particle, a track is born!
                    self.track_cls(dp)
                elif dp is None:
                    # add the unmatched source particles to the new
                    # memory set
                    new_mem_set.add(sp)

                # Clean up
                if dp is not None:
                    del dp.back_cands
                if sp is not None:
                    del sp.forward_cands

            # TODO: Emit debug message with number of
            # subnets in this level, numbers of new/remembered/lost particles

            # yield the current level before we add the memory points to it
            yield cur_level

            # set prev_hash to cur hash
            prev_hash = cur_hash

            # add in the memory points
            # store the current level for use in next loop
            if self.memory > 0:
                # identify the new memory points
                new_mem_set -= self.mem_set
                mem_history.append(new_mem_set)
                # remove points that are now too old
                self.mem_set -= mem_history.pop(0)
                # add the new points
                self.mem_set |= new_mem_set
                # add the memory particles to what will be the next source set
                tmp_set |= self.mem_set
                # add memory points to prev_hash (to be used as the next source)
                for m in self.mem_set:
                    # add points to the hash
                    prev_hash.add_point(m)
                    # Record how many times this particle got "held back".
                    # Since this particle has already been yielded in a previous
                    # level, we can't store it there. We'll have to put it in the
                    # track object, then copy this info to the point in cur_hash
                    # if/when we make a link.
                    m.track.incr_memory()
                    # re-create the forward_cands list
                    m.forward_cands = []

            prev_set = tmp_set

    def _assign_links(self, dest_set, source_set, search_range):
        """Match particles in dest_set with source_set.

        Returns source, dest lists of equal length, corresponding
        to pairs of source and destination particles. A 'None' value
        denotes that a match was not found.

        The contents of dest_set and source_set will be changed, as
        well as the forward_cands and back_cands attributes of the
        particles. However, this does not meaningfully change the state
        within link(). All meaningful actions are taken within link(),
        based on the recommendations of _assign_links().
        """
        spl, dpl = [], []
        diag = self.diag
        # while there are particles left to link, link
        while len(dest_set) > 0:
            p = dest_set.pop()
            bc_c = len(p.back_cands)
            # no backwards candidates
            if bc_c == 0:
                # particle will get a new track
                dpl.append(p)
                spl.append(None)
                if diag:
                    p.diag['search_range'] = search_range
                continue  # do next dest_set particle
            if bc_c == 1:
                # one backwards candidate
                b_c_p = p.back_cands[0]
                # and only one forward candidate
                b_c_p_0 = b_c_p[0]
                if len(b_c_p_0.forward_cands) == 1:
                    # schedule these particles for linking
                    dpl.append(p)
                    spl.append(b_c_p_0)
                    source_set.discard(b_c_p_0)
                    if diag:
                        p.diag['search_range'] = search_range
                    continue  # do next dest_set particle
            # we need to generate the sub networks
            done_flg = False
            s_sn = set()                  # source sub net
            d_sn = set()                  # destination sub net
            # add working particle to destination sub-net
            d_sn.add(p)
            while not done_flg:
                d_sn_sz = len(d_sn)
                s_sn_sz = len(s_sn)
                for dp in d_sn:
                    for c_sp in dp.back_cands:
                        s_sn.add(c_sp[0])
                        source_set.discard(c_sp[0])
                for sp in s_sn:
                    for c_dp in sp.forward_cands:
                        d_sn.add(c_dp[0])
                        dest_set.discard(c_dp[0])
                done_flg = (len(d_sn) == d_sn_sz) and (len(s_sn) == s_sn_sz)

            # sort and add in penalty for not linking
            for _s in s_sn:
                # If we end up having to recurse for adaptive search, this final
                # element will be dropped and re-added, because search_range is
                # decreasing.
                _s.forward_cands.sort(key=lambda x: x[1])
                _s.forward_cands.append((None, search_range))

            try:
                sn_spl, sn_dpl = self.subnet_linker(s_sn, len(d_sn), search_range,
                                                    max_size=self.max_subnet_size,
                                                    diag=diag)

                if diag:
                    # Record information about this invocation of the subnet linker.
                    for dp in d_sn:
                        dp.diag['subnet'] = self.subnet_counter
                        dp.diag['subnet_size'] = len(s_sn)
                        dp.diag['search_range'] = search_range
                for dp in d_sn - set(sn_dpl):
                    # Unclaimed destination particle in subnet
                    sn_spl.append(None)
                    sn_dpl.append(dp)
                self.subnet_counter += 1
            except SubnetOversizeException:
                if self.adaptive_stop is None:
                    raise
                # Reduce search_range
                new_range = search_range * self.adaptive_step
                if search_range <= self.adaptive_stop:
                    # adaptive_stop is the search_range below which linking
                    # is presumed invalid. So we just give up.
                    raise

                # Prune the candidate lists of s_sn, d_sn; then recurse.
                for sp in s_sn:
                    sp.forward_cands = [fc for fc in sp.forward_cands
                                        if fc[1] <= new_range]
                for dp in d_sn:
                    dp.back_cands = [bc for bc in dp.back_cands
                                     if bc[1] <= new_range]
                sn_spl, sn_dpl = self._assign_links(
                    d_sn, s_sn, new_range)

            spl.extend(sn_spl)
            dpl.extend(sn_dpl)

        # Leftovers
        for pp in source_set:
            spl.append(pp)
            dpl.append(None)

        return spl, dpl


def assign_candidates(cur_hash, prev_hash, search_range, neighbor_strategy):
    if neighbor_strategy == 'BTree':
        if not is_isotropic(search_range):
            raise ValueError
        search_range_1d = search_range[0]
        # (Tom's code)
        for p in cur_hash.points:
            work_box = prev_hash.get_region(p, search_range_1d)
            for wp in work_box:
                d = p.distance(wp) / search_range_1d
                if d < 1.:
                    p.back_cands.append((wp, d))
                    wp.forward_cands.append((p, d))
    elif neighbor_strategy == 'KDTree':
        # kdtree.query() would raise exception on empty level.
        if len(cur_hash) and len(prev_hash):
            cur_coords = cur_hash.coords_rescaled
            hashpts = prev_hash.points
            dists, inds = prev_hash.kdtree.query(cur_coords, 10,
                                                 distance_upper_bound=1+1e-7)
            nn = np.sum(np.isfinite(dists), 1)  # Number of neighbors of each particle
            for i, p in enumerate(cur_hash.points):
                for j in range(nn[i]):
                    wp = hashpts[inds[i, j]]
                    p.back_cands.append((wp, dists[i, j]))
                    wp.forward_cands.append((p, dists[i, j]))
