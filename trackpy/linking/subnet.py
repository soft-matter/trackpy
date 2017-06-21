from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
from six.moves import range
import itertools
import functools

import numpy as np
import pandas as pd

from .utils import points_to_arr
from ..utils import default_pos_columns, cKDTree


class TreeFinder(object):
    def __init__(self, points, search_range):
        """Takes a list of particles."""
        self.ndim = len(search_range)
        self.search_range = np.atleast_2d(search_range)
        if not isinstance(points, list):
            points = list(points)
        self.points = points
        self.set_predictor(None)
        self.rebuild()

    def __len__(self):
        return len(self.points)

    def add_point(self, pt):
        self.points.append(pt)
        self._clean = False

    def set_predictor(self, predictor, t=None):
        """Sets a predictor to the TreeFinder

        predictor : function, optional

            Called with t and a list of N Point instances, returns their
            "effective" locations, as an N x d array (or any iterable).
            Used for prediction (see "predict" module).
        """
        if predictor is None:
            self.coord_mapping = functools.partial(_default_coord_mapping,
                                                   self.search_range)
        else:
            self.coord_mapping = _wrap_predictor(self.search_range,
                                                 predictor, t)
        self._clean = False

    @property
    def kdtree(self):
        if not self._clean:
            self.rebuild()
        return self._kdtree

    def rebuild(self):
        """Rebuilds tree from ``points`` attribute.

        coord_map : function, optional

            Called with a list of N Point instances, returns their
            "effective" locations, as an N x d array (or list of tuples).
            Used for prediction (see "predict" module).

        rebuild() needs to be called after ``add_point()`` and
        before tree is used for spatial queries again (i.e. when
        memory is turned on).
        """
        self._clean = False
        if len(self.points) == 0:
            self._kdtree = None
        else:
            coords_mapped = self.coord_mapping(self.points)
            self._kdtree = cKDTree(coords_mapped, 15)
        # This could be tuned
        self._clean = True

    @property
    def coords(self):
        return points_to_arr(self.points)

    @property
    def coords_mapped(self):
        if not self._clean:
            self.rebuild()
        if self._kdtree is None:
            return np.empty((0, self.ndim))
        else:
            return self._kdtree.data

    @property
    def coords_df(self):
        coords = self.coords
        if len(coords) == 0:
            return
        data = pd.DataFrame(coords, columns=default_pos_columns(self.ndim),
                            index=[p.uuid for p in self.points])

        # add placeholders to obtain columns with integer dtype
        data['frame'] = -1
        data['particle'] = -1
        for p in self.points:
            data.loc[p.uuid, 'frame'] = p.t
            data.loc[p.uuid, 'particle'] = p.track.id
            for col in p.extra_data:
                data.loc[p.uuid, col] = p.extra_data[col]
        return data

    def query_points(self, pos, max_dist_normed=1.):
        if self.kdtree is None:
            return
        pos_norm = pos / self.search_range
        found = self.kdtree.query_ball_point(pos_norm, max_dist_normed)
        found = set([i for sl in found for i in sl])  # ravel
        if len(found) == 0:
            return
        else:
            return self.coords[list(found)]


def _default_coord_mapping(search_range, level):
    """ Convert a list of Points to an ndarray of coordinates """
    return points_to_arr(level) / search_range


def _wrap_predictor(search_range, predictor, t):
    """ Create a function that maps coordinates using a predictor class."""
    def coord_mapping(level):
        # swap axes order (need to do inplace to preserve the Point attributes)
        for p in level:
            p.pos = p.pos[::-1]
        result = np.array(list(predictor(t, level)))
        for p in level:  # swap axes order back
            p.pos = p.pos[::-1]
        return result[:, ::-1] / search_range

    return coord_mapping


def assign_subnet(source, dest, subnets):
    """ Assign source point and dest point to the same subnet """
    i1 = source.subnet
    i2 = dest.subnet
    if i1 is None and i2 is None:
        raise ValueError("No subnet for added destination particle")
    if i1 == i2:  # if a and b are already in the same subnet, do nothing
        return
    if i1 is None:  # source did not belong to a subset before
        # just add it
        subnets[i2][0].add(source)
        source.subnet = i2
    elif i2 is None:  # dest did not belong to a subset before
        # just add it
        subnets[i1][1].add(dest)
        dest.subnet = i1
    else:  # source belongs to subset i1 before
        # merge the subnets
        subnets[i2][0].update(subnets[i1][0])
        subnets[i2][1].update(subnets[i1][1])
        # update the subnet identifiers per point
        for p in itertools.chain(*subnets[i1]):
            p.subnet = i2
        # and delete the old source subnet
        del subnets[i1]


def split_subnet(source, dest, new_range):
    # Clear the subnets and candidates for all points in both frames
    subnets = dict()
    for sp in source:
        sp.subnet = None
    for i, dp in enumerate(dest):
        dp.subnet = i
        subnets[i] = set(), {dp}

    for sp in source:
        for dp, dist in sp.forward_cands:
            if dist > new_range:
                continue
            assign_subnet(sp, dp, subnets=subnets)
    return (subnets[key] for key in subnets)


class Subnets(object):
    """ Class that evaluates the possible links between two groups of features.

    Candidates and subnet indices are stored inside the Point objects that are
    inside the provided TreeFinder objects.

    Subnets are based on the destination points: subnets having only a source
    point are not included. They can be accessed from the `lost` method.
    If subnets with only one source point need to be included, call the
    method `include_lost`. In that case, the method `lost` will raise.

    Parameters
    ----------
    source_hash : TreeFinder object
        The hash of the first (source) frame
    dest_hash : TreeFinder object
        The hash of the second (destination) frame
    max_neighbors : int, optional
        The maximum number of linking candidates for one feature. Default 10.

    Attributes
    ----------
    subnets : dictionary
        A dictonary, indexed by subnet index, that contains the subnets as a
        tuple of sets. The first set contains the source points, the second
        set contains the destination points. Iterate over this dictionary by
        directly iterating over the Subnets object.

    Methods
    -------
    get_lost :
        Lists source points without linking candidates ('lost' features).
        Raises if these particles are included in the subnets already, by
        calling `include_lost`.
    """
    def __init__(self, source_hash, dest_hash, max_neighbors=10):
        self.max_neighbors = max_neighbors
        self.source_hash = source_hash
        self.dest_hash = dest_hash
        self.includes_lost = False
        self.reset()
        self.compute()

    def reset(self):
        """ Clear the subnets and candidates for all points in both frames """
        self.subnets = dict()
        for p in self.source_hash.points:
            p.forward_cands = []
            p.subnet = None
        for i, p in enumerate(self.dest_hash.points):
            # p.back_cands = []
            p.subnet = i
            self.subnets[i] = set(), {p}

    def compute(self, search_range=1.):
        """ Evaluate the linking candidates and corresponding subnets, using
        given `search_range` (rescaled to 1.)."""
        source_hash = self.source_hash
        dest_hash = self.dest_hash
        if len(source_hash.points) == 0 or len(dest_hash.points) == 0:
            return
        search_range = float(search_range) + 1e-7
        dists, inds = source_hash.kdtree.query(dest_hash.coords_mapped,
                                               self.max_neighbors,
                                               distance_upper_bound=search_range)
        nn = np.sum(np.isfinite(dists), 1)  # Number of neighbors of each particle
        for i, p in enumerate(dest_hash.points):
            for j in range(nn[i]):
                wp = source_hash.points[inds[i, j]]
                # p.back_cands.append((wp, dists[i, j]))
                wp.forward_cands.append((p, dists[i, j]))
                assign_subnet(wp, p, self.subnets)

    def __iter__(self):
        return (self.subnets[key] for key in self.subnets)

    def lost(self):
        if self.includes_lost:
            raise ValueError('Lost particles are included in the subnets.')
        else:
            return [p for p in self.source_hash.points if p.subnet is None]

    def add_dest_points(self, source_points, dest_points):
        """ Add destination points, evaluate candidates and subnets.

        This code cannot generate new subnets. The given points have to be such
        that new subnets do not have to be created.

        Parameters
        ----------
        source_points : iterable of points
            Consider these points only as linking candidates. They should exist
            already in Subnets.source_points.
        dest_points : iterable of points
            The destination points to add. They should be new.
        """
        # TODO is kdtree really faster here than brute force ?
        if len(dest_points) == 0:
            return
        source_points = list(source_points)
        source_coord = self.source_hash.coord_mapping(source_points)
        new_dest_hash = TreeFinder(dest_points, self.dest_hash.search_range)
        dists, inds = new_dest_hash.kdtree.query(source_coord,
                                                 max(len(source_points), 2),
                                                 distance_upper_bound=1+1e-7)
        nn = np.sum(np.isfinite(dists), 1)  # Number of neighbors of each particle
        for i, source in enumerate(source_points):
            for j in range(nn[i]):
                dest = new_dest_hash.points[inds[i, j]]
                # dest.back_cands.append((source, dists[i, j]))
                source.forward_cands.append((dest, dists[i, j]))
                # source particle always has a subnet, add the dest particle
                self.subnets[source.subnet][1].add(dest)
                dest.subnet = source.subnet

        # sort candidates again because they might have changed
        for p in source_points:
            p.forward_cands.sort(key=lambda x: x[1])
        # for p in dest_hash.points:
        #    p.back_cands.sort(key=lambda x: x[1])

    def include_lost(self):
        """ Add source particles without any destination particle to the
        subnets."""
        if len(self.subnets) > 0:
            counter = itertools.count(start=max(self.subnets) + 1)
        else:
            counter = itertools.count()
        for p in self.source_hash.points:
            if len(p.forward_cands) == 0:
                subnet = next(counter)
                self.subnets[subnet] = {p}, set()
                p.subnet = subnet

        self.includes_lost = True

    def merge_lost_subnets(self):
        """ Merge subnets that have lost features and that are closer than
        twice the search range together, in order to account for the possibility
        that relocated points will join subnets together. """
        if not self.includes_lost:
            self.include_lost()

        # list subnets that have lost particles
        lost_source = []
        for key in self.subnets:
            source, dest = self.subnets[key]
            shortage = len(source) - len(dest)
            if shortage > 0:
                lost_source.extend(source)

        if len(lost_source) == 0:
            return
        lost_coords = self.source_hash.coord_mapping(lost_source)
        dists, inds = self.source_hash.kdtree.query(lost_coords, self.max_neighbors,
                                                    distance_upper_bound=2+1e-7)
        nn = np.sum(np.isfinite(dists), 1)  # Number of neighbors of each particle
        for i, p in enumerate(lost_source):
            for j in range(nn[i]):
                wp = self.source_hash.points[inds[i, j]]
                i1, i2 = p.subnet, wp.subnet
                if i1 != i2:
                    if i2 > i1:
                        i1, i2 = i2, i1
                    self.subnets[i2][0].update(self.subnets[i1][0])
                    self.subnets[i2][1].update(self.subnets[i1][1])
                    # update the subnet identifiers per point
                    for p in itertools.chain(*self.subnets[i1]):
                        p.subnet = i2
                    # and delete the old source subnet
                    del self.subnets[i1]
