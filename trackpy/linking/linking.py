import warnings
import logging
import itertools, functools

import numpy as np

from ..utils import (guess_pos_columns, default_pos_columns,
                     validate_tuple, is_isotropic, pandas_sort)
from ..try_numba import NUMBA_AVAILABLE
from .utils import (Point, TrackUnstored, points_from_arr,
                    coords_from_df, coords_from_df_iter,
                    SubnetOversizeException)
from .subnet import HashBTree, HashKDTree, Subnets, split_subnet
from .subnetlinker import (subnet_linker_recursive, subnet_linker_drop,
                           subnet_linker_numba, subnet_linker_nonrecursive)

logger = logging.getLogger(__name__)


def link_iter(coords_iter, search_range, **kwargs):
    """
    link_iter(coords_iter, search_range, memory=0, predictor=None,
        adaptive_stop=None, adaptive_step=0.95, neighbor_strategy=None,
        link_strategy=None, dist_func=None, to_eucl=None)

    Link an iterable of per-frame coordinates into trajectories.

    Parameters
    ----------
    coords_iter : iterable
        the iterable produces 2d numpy arrays of coordinates (shape: N, ndim).
        to tell link_iter what frame number each array is, the iterable may
        be enumerated so that it produces (number, 2d array) tuples
    search_range : float or tuple
        the maximum distance features can move between frames,
        optionally per dimension
    memory : integer, optional
        the maximum number of frames during which a feature can vanish,
        then reappear nearby, and be considered the same particle. Default: 0
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
    neighbor_strategy : {'KDTree', 'BTree'}
        algorithm used to identify nearby features. Default 'KDTree'.
    link_strategy : {'recursive', 'nonrecursive', 'hybrid', 'numba', 'drop', 'auto'}
        algorithm used to resolve subnetworks of nearby particles
        'auto' uses hybrid (numba+recursive) if available
        'drop' causes particles in subnetworks to go unlinked
    dist_func : function or ```sklearn.metrics.DistanceMetric``` instance, optional
        A custom python distance function or instance of the 
        Scikit Learn DistanceMetric class. If a python distance function is 
        passed, it must take two 1D arrays of coordinates and return a float. 
        Must be used with the 'BTree' neighbor_strategy.
    to_eucl : function, optional
        function that transforms a N x ndim array of positions into coordinates
        in Euclidean space. Useful for instance to link by Euclidean distance
        starting from radial coordinates. If search_range is anisotropic, this
        parameter cannot be used.

    Yields
    ------
    tuples (t, list of particle ids)

    See also
    --------
    link

    Notes
    -----
    This is an implementation of the Crocker-Grier linking algorithm.
    [1]_

    References
    ----------
    .. [1] Crocker, J.C., Grier, D.G. http://dx.doi.org/10.1006/jcis.1996.0217
    """
    # ensure that coords_iter is iterable
    coords_iter = iter(coords_iter)

    # interpret the first element of the iterable
    val = next(coords_iter)
    if isinstance(val, np.ndarray):
        # the iterable was not enumerated, so enumerate the remainder
        coords_iter = enumerate(coords_iter, start=1)
        t, coords = 0, val
    else:
        t, coords = val

    # initialize the linker and yield the particle ids of the first frame
    linker = Linker(search_range, **kwargs)
    linker.init_level(coords, t)
    yield t, linker.particle_ids

    for t, coords in coords_iter:
        linker.next_level(coords, t)
        logger.info("Frame {}: {} trajectories present.".format(t, len(linker.particle_ids)))
        yield t, linker.particle_ids


def link(f, search_range, pos_columns=None, t_column='frame', **kwargs):
    """
    link(f, search_range, pos_columns=None, t_column='frame', memory=0,
        predictor=None, adaptive_stop=None, adaptive_step=0.95,
        neighbor_strategy=None, link_strategy=None, dist_func=None,
        to_eucl=None)

    Link a DataFrame of coordinates into trajectories.

    Parameters
    ----------
    f : DataFrame
        The DataFrame must include any number of column(s) for position and a
        column of frame numbers. By default, 'x' and 'y' are expected for
        position, and 'frame' is expected for frame number. See below for
        options to use custom column names.
    search_range : float or tuple
        the maximum distance features can move between frames,
        optionally per dimension
    pos_columns : list of str, optional
        Default is ['y', 'x'], or ['z', 'y', 'x'] when 'z' is present in f
    t_column : str, optional
        Default is 'frame'
    memory : integer, optional
        the maximum number of frames during which a feature can vanish,
        then reappear nearby, and be considered the same particle. 0 by default.
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
    neighbor_strategy : {'KDTree', 'BTree'}
        algorithm used to identify nearby features. Default 'KDTree'.
    link_strategy : {'recursive', 'nonrecursive', 'numba', 'hybrid', 'drop', 'auto'}
        algorithm used to resolve subnetworks of nearby particles
        'auto' uses hybrid (numba+recursive) if available
        'drop' causes particles in subnetworks to go unlinked
    dist_func : function or ```sklearn.metrics.DistanceMetric``` instance, optional
        A custom python distance function or instance of the 
        Scikit Learn DistanceMetric class. If a python distance function is 
        passed, it must take two 1D arrays of coordinates and return a float. 
        Must be used with the 'BTree' neighbor_strategy.
    to_eucl : function, optional
        function that transforms a N x ndim array of positions into coordinates
        in Euclidean space. Useful for instance to link by Euclidean distance
        starting from radial coordinates. If search_range is anisotropic, this
        parameter cannot be used.

    Returns
    -------
    DataFrame with added column 'particle' containing trajectory labels.
    The t_column (by default: 'frame') will be coerced to integer.

    See also
    --------
    link_iter

    Notes
    -----
    This is an implementation of the Crocker-Grier linking algorithm.
    [1]_

    References
    ----------
    .. [1] Crocker, J.C., Grier, D.G. http://dx.doi.org/10.1006/jcis.1996.0217

    """
    if pos_columns is None:
        pos_columns = guess_pos_columns(f)

    # copy the dataframe
    f = f.copy()
    # coerce t_column to integer type (use np.int64 to avoid 32-bit on windows)
    if not np.issubdtype(f[t_column].dtype, np.integer):
        f[t_column] = f[t_column].astype(np.int64)
    # sort on the t_column
    pandas_sort(f, t_column, inplace=True)

    coords_iter = coords_from_df(f, pos_columns, t_column)
    ids = []
    for i, _ids in link_iter(coords_iter, search_range, **kwargs):
        ids.extend(_ids)

    f['particle'] = ids
    return f

link_df = link


def link_df_iter(f_iter, search_range, pos_columns=None,
                 t_column='frame', **kwargs):
    """
    link_df_iter(f_iter, search_range, pos_columns=None, t_column='frame',
        memory=0, predictor=None, adaptive_stop=None, adaptive_step=0.95,
        neighbor_strategy=None, link_strategy=None, dist_func=None,
        to_eucl=None)

    Link an iterable of DataFrames into trajectories.

    Parameters
    ----------
    f_iter : iterable of DataFrames
        Each DataFrame must include any number of column(s) for position and a
        column of frame numbers. By default, 'x' and 'y' are expected for
        position, and 'frame' is expected for frame number. For optimal
        performance, explicitly specify the column names using `pos_columns`
        and `t_column` kwargs.
    search_range : float or tuple
        the maximum distance features can move between frames,
        optionally per dimension
    pos_columns : list of str, optional
        Default is ['y', 'x'], or ['z', 'y', 'x'] when 'z' is present in f
        If this is not supplied, f_iter will be investigated, which might cost
        performance. For optimal performance, always supply this parameter.
    t_column : str, optional
        Default is 'frame'
    memory : integer, optional
        the maximum number of frames during which a feature can vanish,
        then reappear nearby, and be considered the same particle. 0 by default.
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
    neighbor_strategy : {'KDTree', 'BTree'}
        algorithm used to identify nearby features. Default 'KDTree'.
    link_strategy : {'recursive', 'nonrecursive', 'numba', 'hybrid', 'drop', 'auto'}
        algorithm used to resolve subnetworks of nearby particles
        'auto' uses hybrid (numba+recursive) if available
        'drop' causes particles in subnetworks to go unlinked
    dist_func : function or ```sklearn.metrics.DistanceMetric``` instance, optional
        A custom python distance function or instance of the 
        Scikit Learn DistanceMetric class. If a python distance function is 
        passed, it must take two 1D arrays of coordinates and return a float. 
        Must be used with the 'BTree' neighbor_strategy.
    to_eucl : function, optional
        function that transforms a N x ndim array of positions into coordinates
        in Euclidean space. Useful for instance to link by Euclidean distance
        starting from radial coordinates. If search_range is anisotropic, this
        parameter cannot be used.

    Yields
    ------
    DataFrames with added column 'particle' containing trajectory labels

    See also
    --------
    link
    """
    if pos_columns is None:
        # Get info about the first frame without processing it
        f_iter, f_iter_dummy = itertools.tee(f_iter)
        f0 = next(f_iter_dummy)
        pos_columns = guess_pos_columns(f0)
        del f_iter_dummy, f0

    f_iter, f_coords_iter = itertools.tee(f_iter)
    coords_iter = coords_from_df_iter(f_coords_iter, pos_columns, t_column)

    ids_iter = (_ids for _i, _ids in
        link_iter(coords_iter, search_range, **kwargs))
    for df, ids in zip(f_iter, ids_iter):
        df_linked = df.copy()
        df_linked['particle'] = ids
        yield df_linked


def adaptive_link_wrap(source_set, dest_set, search_range, subnet_linker,
                       adaptive_stop=None, adaptive_step=0.95, **kwargs):
    """Wraps a subnetlinker, making it adaptive."""
    try:
        sn_spl, sn_dpl = subnet_linker(source_set, dest_set,
                                       search_range, **kwargs)
    except SubnetOversizeException:
        if adaptive_stop is None:
            raise
        new_range = search_range * adaptive_step
        if search_range <= adaptive_stop:
            # adaptive_stop is the search_range below which linking
            # is presumed invalid. So we just give up.
            raise

        # Split the subnet and recurse
        sn_spl = []
        sn_dpl = []
        for source, dest in split_subnet(source_set, dest_set, new_range):
            split_spl, split_dpl = \
                adaptive_link_wrap(source, dest, new_range, subnet_linker,
                                   adaptive_stop, adaptive_step, **kwargs)
            sn_spl.extend(split_spl)
            sn_dpl.extend(split_dpl)

    return sn_spl, sn_dpl


class Linker:
    """Linker class that sequentially links ndarrays of coordinates together.

    The class can be used via the `init_level` and `next_level` methods.

    Attributes
    ----------
    hash : Hash object
        The hash containing the points of the current level
    mem_set : set of points
    mem_history : list of sets of points
    particle_ids : list
        a list of track ids of the current hash
    points : list of points
        The points of the current hash.
    coords : ndarray
        The coordinates of the points of the current hash. It is possible to
        write on this attribute if the number of coordinates stays constant.
    coords_df : DataFrame
        The coordinates of the points of the current hash. It is possible to
        write on this attribute, changing positional coordinates only, if the
        number of coordinates stays constant.
    subnets : Subnets
        Subnets object containing the subnets of the prev and current points.

    Methods
    -------
    init_level(coords, t, extra_data)
        creates the first level (frame): no linking is done
    next_level(coords, t, extra_data)
        Add a level, assign candidates and subnets, and apply the links.
    update_hash(coords, t, extra_data)
        Updates the hash: the previous hash is returned
    assign_links()
        Assign links between previous and current points (given by obj.subnets)
        Returns a list of source particles and a list of destination particles
        that are to be linked.
    apply_links(spl, dpl)
        Applies links between the source particle list (spl) and destination
        particle list (dpl)

    See also
    --------
    link
    """
    # Largest subnet we will attempt to solve.
    MAX_SUB_NET_SIZE = 30
    # For adaptive search, subnet linking should fail much faster.
    MAX_SUB_NET_SIZE_ADAPTIVE = 15
    # Maximum number of candidates per particle
    MAX_NEIGHBORS = 10

    def __init__(self, search_range, memory=0, predictor=None,
                 adaptive_stop=None, adaptive_step=0.95,
                 neighbor_strategy=None, link_strategy=None,
                 dist_func=None, to_eucl=None):
        self.memory = memory
        self.predictor = predictor
        self.track_cls = TrackUnstored
        self.ndim = None  # unknown at this point; inferred at init_level()

        if neighbor_strategy is None:
            if dist_func is None:
                neighbor_strategy = 'KDTree'
            else:
                neighbor_strategy = 'BTree'
        elif neighbor_strategy not in ['KDTree', 'BTree']:
            raise ValueError("neighbor_strategy must be 'KDTree' or 'BTree'")
        elif neighbor_strategy != 'BTree' and dist_func is not None:
            raise ValueError("For custom distance functions please use "
                             "the 'BTree' neighbor_strategy.")

        self.hash_cls = dict(KDTree=HashKDTree,
                             BTree=HashBTree)[neighbor_strategy]
        self.dist_func = dist_func  # a custom distance function
        self.to_eucl = to_eucl      # to euclidean coordinates

        if link_strategy is None or link_strategy == 'auto':
            if NUMBA_AVAILABLE:
                link_strategy = 'hybrid'
            else:
                link_strategy = 'recursive'

        if link_strategy == 'recursive':
            subnet_linker = subnet_linker_recursive
        elif link_strategy == 'hybrid':
            subnet_linker = subnet_linker_numba
        elif link_strategy == 'numba':
            subnet_linker = functools.partial(subnet_linker_numba,
                                              hybrid=False)
        elif link_strategy == 'nonrecursive':
            subnet_linker = subnet_linker_nonrecursive
        elif link_strategy == 'drop':
            subnet_linker = subnet_linker_drop
        elif callable(link_strategy):
            subnet_linker = link_strategy
        else:
            raise ValueError("Unknown linking strategy '{}'".format(link_strategy))

        # if search_range is anisotropic, transform coordinates to a rescaled
        # space with search_range == 1.
        if is_isotropic(search_range):
            if hasattr(search_range, '__iter__'):
                self.search_range = float(search_range[0])
            else:
                self.search_range = float(search_range)
        elif self.to_eucl is not None:
            raise ValueError('Cannot use anisotropic search ranges in '
                             'combination with a coordinate transformation.')
        else:
            search_range = np.atleast_2d(search_range)
            self.to_eucl = lambda x: x / search_range
            self.search_range = 1.
            # also rescale adaptive_stop
            if adaptive_stop is not None:
                adaptive_stop = np.max(adaptive_stop / search_range)

        self.hash = None
        self.mem_set = set()

        if adaptive_stop is not None:
            if 1 * adaptive_stop <= 0:
                raise ValueError("adaptive_stop must be positive.")
            self.subnet_linker = functools.partial(adaptive_link_wrap,
                                                   subnet_linker=subnet_linker,
                                                   adaptive_stop=adaptive_stop,
                                                   adaptive_step=adaptive_step,
                                                   max_size=self.MAX_SUB_NET_SIZE_ADAPTIVE)
        else:
            self.subnet_linker = functools.partial(subnet_linker,
                                                   max_size=self.MAX_SUB_NET_SIZE)

    def update_hash(self, coords, t, extra_data=None):
        if self.ndim is None:
            raise RuntimeError("The Linker was not initialized. Please use "
                               "`init_level` for the first level.")
        prev_hash = self.hash
        # add memory points to prev_hash (to be used as the next source)
        for m in self.mem_set:
            # add points to the hash
            prev_hash.add_point(m)
            # re-create the forward_cands list
            m.forward_cands = []

        # If prediction is enabled, we need to update the positions in prev_hash
        # to where we think they'll be in the frame corresponding to 'coords'.
        if prev_hash is not None and self.predictor is not None:
            prev_hash.set_predictor(self.predictor, t)  # Rewrite positions

        self.hash = self.hash_cls(points_from_arr(coords, t, extra_data),
                                  ndim=self.ndim, to_eucl=self.to_eucl,
                                  dist_func=self.dist_func)
        return prev_hash

    def init_level(self, coords, t, extra_data=None):
        Point.reset_counter()
        TrackUnstored.reset_counter()
        self.ndim = coords.shape[1]
        self.mem_set = set()
        # Initialize memory with empty sets.
        self.mem_history = []
        for j in range(self.memory):
            self.mem_history.append(set())

        self.update_hash(coords, t, extra_data)
        # Assume everything in first level starts a Track.
        # Iterate over prev_level, not prev_set, because order -> track ID.
        for p in self.hash.points:
            TrackUnstored(p)

    @property
    def particle_ids(self):
        return [p.track.id for p in self.hash.points]

    @property
    def coords(self):
        return self.hash.coords
    @coords.setter
    def coords(self, value):
        if len(value) != len(self.hash.points):
            raise ValueError("Number of coordinates has changed")
        for coord, pnt in zip(value, self.hash.points):
            pnt.pos = coord
        self.hash._clean = False

    @property
    def coords_df(self):
        return self.hash.coords_df
    @coords_df.setter
    def coords_df(self, value):
        if len(value) != len(self.hash.points):
            raise ValueError("Number of features has changed")
        self.coords = value[default_pos_columns(self.ndim)].values

    def next_level(self, coords, t, extra_data=None):
        prev_hash = self.update_hash(coords, t, extra_data)

        self.subnets = Subnets(prev_hash, self.hash, self.search_range,
                               self.MAX_NEIGHBORS)
        spl, dpl = self.assign_links()
        self.apply_links(spl, dpl)

    def assign_links(self):
        spl, dpl = [], []
        for source_set, dest_set in self.subnets:
            for sp in source_set:
                sp.forward_cands.sort(key=lambda x: x[1])

            sn_spl, sn_dpl = self.subnet_linker(source_set, dest_set,
                                                self.search_range)
            spl.extend(sn_spl)
            dpl.extend(sn_dpl)

        # Leftovers
        lost = self.subnets.lost
        spl.extend(lost)
        dpl.extend([None] * len(lost))

        return spl, dpl

    def apply_links(self, spl, dpl):
        new_mem_set = set()
        for sp, dp in zip(spl, dpl):
            # Do linking
            if sp is not None and dp is not None:
                sp.track.add_point(dp)
                if sp in self.mem_set:  # Very rare
                    self.mem_set.remove(sp)
            elif sp is None:
                # if unclaimed destination particle, a track is born!
                TrackUnstored(dp)
            elif dp is None:
                # add the unmatched source particles to the new
                # memory set
                new_mem_set.add(sp)

            # Clean up
            if sp is not None:
                sp.forward_cands = []

        # add in the memory points
        # store the current level for use in next loop
        if self.memory > 0:
            # identify the new memory points
            new_mem_set -= self.mem_set
            self.mem_history.append(new_mem_set)
            # remove points that are now too old
            self.mem_set -= self.mem_history.pop(0)
            # add the new points
            self.mem_set |= new_mem_set
