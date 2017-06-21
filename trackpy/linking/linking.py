from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
from six.moves import range, zip
import warnings
import logging
import itertools, functools

import numpy as np

from ..utils import (default_pos_columns, guess_pos_columns,
                     validate_tuple, pandas_sort)
from ..try_numba import NUMBA_AVAILABLE
from .utils import (Point, TrackUnstored, points_from_arr,
                    coords_from_df, coords_from_df_iter,
                    SubnetOversizeException)
from .subnet import TreeFinder, Subnets, split_subnet
from .subnetlinker import (subnet_linker_recursive, subnet_linker_drop,
                           subnet_linker_numba, subnet_linker_nonrecursive)

logger = logging.getLogger(__name__)


def link_iter(coords_iter, search_range, **kwargs):
    """Link an iterable of per-frame coordinates into trajectories.

    Parameters
    ----------
    coords_iter : iterable or enumerated iterable of 2d numpy arrays
    search_range : float or tuple
    memory : integer
    predictor : predictor function; see 'predict' module

    Returns
    -------
    yields tuples (t, list of particle ids)
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

    #  obtain dimensionality
    ndim = coords.shape[1]
    search_range = validate_tuple(search_range, ndim)

    # initialize the linker and yield the particle ids of the first frame
    linker = Linker(search_range, **kwargs)
    linker.init_level(coords, t)
    yield t, linker.particle_ids

    for t, coords in coords_iter:
        linker.next_level(coords, t)
        yield t, linker.particle_ids


def link(f, search_range, pos_columns=None, t_column='frame', **kwargs):
    """Link a DataFrame of coordinates into trajectories.

    Parameters
    ----------
    f : DataFrame containing feature positions and frame indices
    search_range : float or tuple
    memory : integer, optional
    pos_columns : list of str, optional
    t_column : str, optional

    Returns
    -------
    DataFrame with added column 'particle' containing trajectory labels.
    The t_column (by default: 'frame') will be coerced to integer."""
    if pos_columns is None:
        pos_columns = guess_pos_columns(f)
    ndim = len(pos_columns)
    search_range = validate_tuple(search_range, ndim)

    # copy the dataframe
    f = f.copy()
    # coerce t_column to integer type
    if not np.issubdtype(f[t_column].dtype, np.integer):
        f[t_column] = f[t_column].astype(np.integer)
    # sort on the t_column
    pandas_sort(f, t_column, inplace=True)

    coords_iter = coords_from_df(f, pos_columns, t_column)
    ids = []
    for i, _ids in link_iter(coords_iter, search_range, **kwargs):
        ids.extend(_ids)

    f['particle'] = ids
    return f


def link_df_iter(f_iter, search_range, pos_columns=None,
                 t_column='frame', **kwargs):
    """Link an iterable of DataFrames into trajectories.

    Parameters
    ----------
    f_iter : iterable of DataFrames with feature positions, frame indices
    search_range : float or tuple
    memory : integer, optional
    pos_columns : list of str, optional
    t_column : str, optional
    predictor : predictor function; see 'predict' module

    Yields
    -------
    DataFrames with added column 'particle' containing trajectory labels
    """
    if pos_columns is None:
        # Get info about the first frame without processing it
        f_iter, f_iter_dummy = itertools.tee(f_iter)
        f0 = next(f_iter_dummy)
        pos_columns = guess_pos_columns(f0)
        del f_iter_dummy, f0
    ndim = len(pos_columns)
    search_range = validate_tuple(search_range, ndim)

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


def _sort_key_spl_dpl(x):
    if x[0] is not None:
        return list(x[0].pos)
    else:
        return list(x[1].pos)


class Linker(object):
    """ Re-implementation of trackpy.linking.Linker for use in find_link.

    Attributes
    ----------
    hash : TreeFinder
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
    """
    # Largest subnet we will attempt to solve.
    MAX_SUB_NET_SIZE = 30
    # For adaptive search, subnet linking should fail much faster.
    MAX_SUB_NET_SIZE_ADAPTIVE = 15
    # Maximum number of candidates per particle
    MAX_NEIGHBORS = 10

    def __init__(self, search_range, memory=0, link_strategy=None,
                 predictor=None, adaptive_stop=None, adaptive_step=0.95):
        self.memory = memory
        self.predictor = predictor
        self.track_cls = TrackUnstored
        self.adaptive_stop = adaptive_stop
        self.adaptive_step = adaptive_step

        if link_strategy is None or link_strategy == 'auto':
            if NUMBA_AVAILABLE:
                link_strategy = 'numba'
            else:
                link_strategy = 'recursive'

        if link_strategy == 'recursive':
            subnet_linker = subnet_linker_recursive
        elif link_strategy == 'numba':
            subnet_linker = subnet_linker_numba
        elif link_strategy == 'nonrecursive':
            subnet_linker = subnet_linker_nonrecursive
        elif link_strategy == 'drop':
            subnet_linker = subnet_linker_drop
        elif callable(link_strategy):
            subnet_linker = link_strategy
        else:
            raise ValueError("Unknown linking strategy '{}'".format(link_strategy))

        self.ndim = len(search_range)
        self.search_range = np.array(search_range)
        self.hash = None
        self.mem_set = set()

        if self.adaptive_stop is not None:
            # internal adaptive_stop is a fraction of search range
            adaptive_stop = np.max(adaptive_stop / self.search_range)
            if 1 * self.adaptive_stop <= 0:
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
        prev_hash = self.hash
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

        # If prediction is enabled, we need to update the positions in prev_hash
        # to where we think they'll be in the frame corresponding to 'coords'.
        if prev_hash is not None and self.predictor is not None:
            prev_hash.set_predictor(self.predictor, t)  # Rewrite positions

        self.hash = TreeFinder(points_from_arr(coords, t, extra_data),
                               self.search_range)
        return prev_hash

    def init_level(self, coords, t, extra_data=None):
        Point.reset_counter()
        TrackUnstored.reset_counter()
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

        self.subnets = Subnets(prev_hash, self.hash, self.MAX_NEIGHBORS)
        spl, dpl = self.assign_links()
        self.apply_links(spl, dpl)

    def assign_links(self):
        spl, dpl = [], []
        for source_set, dest_set in self.subnets:
            sn_spl, sn_dpl = self.subnet_linker(source_set, dest_set, 1.)
            spl.extend(sn_spl)
            dpl.extend(sn_dpl)

        # Leftovers
        lost = self.subnets.lost()
        spl.extend(lost)
        dpl.extend([None] * len(lost))

        return spl, dpl

    def apply_links(self, spl, dpl):
        new_mem_set = set()
        for sp, dp in sorted(zip(spl, dpl), key=_sort_key_spl_dpl):
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

            # # Clean up
            # if dp is not None:
            #     dp.back_cands = []
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
