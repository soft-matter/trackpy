import itertools
import functools

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .utils import points_to_arr
from ..utils import default_pos_columns

try:
    from sklearn.neighbors import BallTree
    try:
        from sklearn.metrics import DistanceMetric
    except ImportError:
        from sklearn.neighbors import DistanceMetric
except ImportError:
    BallTree = None


class HashBase:
    """ Base for classes that efficiently find features near a point. """
    def __init__(self, points, ndim):
        """Takes a list of particles."""
        self.ndim = ndim
        if not isinstance(points, list):
            points = list(points)
        self.points = points
        self.set_predictor(None)

    def __len__(self):
        return len(self.points)

    def add_point(self, pt):
        self.points.append(pt)
        self._clean = False

    def set_predictor(self, predictor, t=None):
        """Sets a predictor to the Hash

        predictor : function, optional

            Called with t and a list of N Point instances, returns their
            "effective" locations, as an N x d array (or any iterable).
            Used for prediction (see "predict" module).
        """
        self.t = t
        self.predictor = predictor
        self._clean = False

    def predict(self, points):
        """Predict and convert points to an array."""
        if self.predictor is None:
            return points_to_arr(points)
        return np.array(list(self.predictor(self.t, points)))

    @property
    def coords(self):
        return points_to_arr(self.points)

    @property
    def coords_predict(self):
        """ Maps coordinates using a predictor class."""
        if self.predictor is None:
            return self.coords
        else:
            return self.predict(self.points)

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


class HashKDTree(HashBase):
    """ Implementation of hash using scipy.spatial.cKDTree """
    def __init__(self, points, ndim, to_eucl=None, dist_func=None):
        """Takes a list of particles."""
        if dist_func is not None:
            raise ValueError("For custom distance functions please use "
                             "the 'BTree' neighbor_strategy.")
        super().__init__(points, ndim)
        if to_eucl is None:
            self.to_eucl = lambda x: x
        else:
            self.to_eucl = to_eucl

    @property
    def tree(self):
        if not self._clean:
            self.rebuild()
        return self._kdtree

    @property
    def coords_mapped(self):
        if not self._clean:
            self.rebuild()
        if self._kdtree is None:
            return np.empty((0, self.ndim))
        else:
            return self._kdtree.data

    def rebuild(self):
        """Rebuilds tree from ``points`` attribute."""
        self._clean = False
        if len(self.points) == 0:
            self._kdtree = None
        else:
            coords_mapped = self.to_eucl(self.coords_predict)
            self._kdtree = cKDTree(coords_mapped, 15)
        # This could be tuned
        self._clean = True

    def query(self, pos, max_neighbors, search_range, rescale=True):
        """Find `max_neighbors` nearest neighbors of `pos` in the hash, with a
        maximum distance of `search_range`. `rescale` determines whether `pos`
        will be rescaled to internal hash coordinates."""
        if self.tree is None:
            return
        if rescale:
            pos = self.to_eucl(pos)
        return self.tree.query(pos, max_neighbors,
                               distance_upper_bound=search_range + 1e-7)

    def query_points(self, pos, search_range, rescale=True):
        """Find the nearest neighbors of `pos` in the hash, with a maximum
        distance of `search_range`. `rescale` determines whether `pos` will
        be rescaled to internal hash coordinates."""
        if self.tree is None:
            return
        if rescale:
            pos = self.to_eucl(pos)
        found = self.tree.query_ball_point(pos, search_range)
        found = {i for sl in found for i in sl}  # ravel
        if len(found) == 0:
            return
        else:
            return self.coords[list(found)]


class HashBTree(HashBase):
    """ Implementation of hash using sklearn.neighbors.BallTree """
    def __init__(self, points, ndim, to_eucl=None, dist_func=None):
        """Takes a list of particles."""
        if BallTree is None:
            raise ImportError("Scikit-learn (sklearn) is required "
                              "for using the 'BTree' neighbor_strategy.")
        super().__init__(points, ndim)
        if to_eucl is None:
            self.to_eucl = lambda x: x
        else:
            self.to_eucl = to_eucl
        self.dist_func = dist_func
        self.rebuild()

    @property
    def btree(self):
        if not self._clean:
            self.rebuild()
        return self._btree

    @property
    def coords_mapped(self):
        if not self._clean:
            self.rebuild()
        if self._btree is None:
            return np.empty((0, self.ndim))
        else:
            return self._btree.data

    def rebuild(self):
        """Rebuilds tree from ``points`` attribute."""
        self._clean = False
        if len(self.points) == 0:
            self._btree = None
        else:
            coords_mapped = self.to_eucl(self.coords_predict)

            if self.dist_func is None:
                self._btree = BallTree(coords_mapped)
            else:
                if isinstance(self.dist_func, DistanceMetric):
                    self._btree = BallTree(coords_mapped,
                                           metric=self.dist_func)
                else:
                    self._btree = BallTree(coords_mapped,
                                           metric='pyfunc', func=self.dist_func)
        # This could be tuned
        self._clean = True

    def query(self, pos, max_neighbors, search_range, rescale=True):
        """Find `max_neighbors` nearest neighbors of `pos` in the hash, with a
        maximum distance of `search_range`. `rescale` determines whether `pos`
        will be rescaled to internal hash coordinates."""
        if self.btree is None:
            return
        if rescale:
            pos = self.to_eucl(pos)
        if max_neighbors > len(self):
            max_neighbors = len(self)
        dists, inds = self.btree.query(pos, k=max_neighbors)
        mask = dists > search_range
        dists[mask] = np.inf
        inds[mask] = len(pos) + 1
        return dists, inds

    def query_points(self, pos, search_range, rescale=True):
        """Find the nearest neighbors of `pos` in the hash, with a maximum
        distance of `search_range`. `rescale` determines whether `pos` will
        be rescaled to internal hash coordinates."""
        if self.btree is None:
            return
        if rescale:
            pos = self.to_eucl(pos)
        dists, found = self.btree.query(pos, return_distance=True)
        found = set(found[dists <= search_range])
        if len(found) == 0:
            return
        else:
            return self.coords[list(found)]


def assign_subnet(source, dest, subnets):
    """ Assign source point and dest point to the same subnet """
    i1 = source.subnet
    i2 = dest.subnet
    if i1 is None and i2 is None:
        raise ValueError("No subnet for added destination particle")
    if i1 == i2:  # if a and b are already in the same subnet, do nothing
        return
    if i1 is None:  # source did not belong to a subnet before
        # just add it
        subnets[i2][0].add(source)
        source.subnet = i2
    elif i2 is None:  # dest did not belong to a subnet before
        # just add it
        subnets[i1][1].add(dest)
        dest.subnet = i1
    else:  # source belongs to subnet i1 before
        # merge the subnets
        subnets[i2][0].update(subnets[i1][0])
        subnets[i2][1].update(subnets[i1][1])
        # update the subnet identifiers per point
        for p in itertools.chain(*subnets[i1]):
            p.subnet = i2
        # and delete the old source subnet
        del subnets[i1]


def split_subnet(source, dest, new_range):
    """Break apart a subnet by using a reduced search_range."""
    subnets = dict()
    # Each destination particle gets its own fresh subnet.
    # These are numbered differently from the "global" dictionary
    # of subnets maintained by the instance of the Subnet class.
    for i, dp in enumerate(dest):
        dp.subnet = i
        subnets[i] = set(), {dp}
    # Clear source particles' subnets, and prune their forward candidates
    # according to new_range.
    # The pruning step is crucial because some subnet linkers ignore
    # the destination set and use the source particles' forward candidates
    # exclusively.
    for sp in source:
        sp.subnet = None
        new_fcs = []
        for dp, dist in sp.forward_cands:
            # Remove particles that are outside new_range
            # (including, presumably, the null candidate)
            if dist <= new_range:
                new_fcs.append((dp, dist))
            else:
                break  # List was sorted by distance
        # There's no need to re-add the null candidate here; that will be done by the
        # subnet linker if needed
        # new_fcs.append((None, new_range))
        sp.forward_cands = new_fcs

        for dp, dist in new_fcs:
            # Null candidates were removed
            # if dp is None:
            #     continue
            assign_subnet(sp, dp, subnets=subnets)
    return (subnets[key] for key in subnets)


class Subnets:
    """ Class that identifies the possible links between two groups of features.

    Candidates and subnet indices are stored inside the Point objects that are
    inside the provided TreeFinder objects.

    Subnets are based on the destination points: subnets having only a source
    point are not included. They can be accessed from the `lost` method.
    If subnets with only one source point need to be included, call the
    method `include_lost`. In that case, the method `lost` will raise.

    In general, subnets computed by this class need to be further evaluated
    to determine the best way to link the source and destination features.

    Parameters
    ----------
    source_hash : TreeFinder object
        The hash of the first (source) frame
    dest_hash : TreeFinder object
        The hash of the second (destination) frame
    search_range : float
    max_neighbors : int, optional
        The maximum number of linking candidates for one feature. Default 10.

    Attributes
    ----------
    subnets : dictionary
        A dictonary, indexed by subnet index, that contains the subnets as a
        tuple of sets. The first set contains the source points, the second
        set contains the destination points. Iterate over this dictionary by
        directly iterating over the Subnets object.
    lost : list
        Lists source points without linking candidates ('lost' features).
        Raises if `includes_lost` == True.
    includes_lost : Boolean
        Whether the source points without linking candidates are included.
    """
    def __init__(self, source_hash, dest_hash, search_range, max_neighbors=10):
        self.max_neighbors = max_neighbors
        self.source_hash = source_hash
        self.dest_hash = dest_hash
        self.search_range = search_range
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
            p.subnet = i
            self.subnets[i] = set(), {p}

    def compute(self):
        """ Evaluate the linking candidates and corresponding subnets, using
        given `search_range`."""
        source_hash = self.source_hash
        dest_hash = self.dest_hash
        if len(source_hash.points) == 0 or len(dest_hash.points) == 0:
            return
        dists, inds = source_hash.query(dest_hash.coords_mapped,
                                        self.max_neighbors, rescale=False,
                                        search_range=self.search_range)
        nn = np.sum(np.isfinite(dists), 1)  # Number of neighbors of each particle
        for i, p in enumerate(dest_hash.points):
            for j in range(nn[i]):
                wp = source_hash.points[inds[i, j]]
                wp.forward_cands.append((p, dists[i, j]))
                assign_subnet(wp, p, self.subnets)

    def __iter__(self):
        return (self.subnets[key] for key in self.subnets)

    @property
    def lost(self):
        if self.includes_lost:
            raise ValueError('Lost particles are included in the subnets.')
        else:
            return [p for p in self.source_hash.points if p.subnet is None]

    def add_dest_points(self, source_points, dest_points, search_range):
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
        source_hash = self.source_hash
        source_coord = source_hash.predict(source_points)
        new_dest_hash = source_hash.__class__(dest_points, search_range)
        dists, inds = new_dest_hash.query(source_coord,
                                          max(len(source_points), 2),
                                          rescale=True,
                                          search_range=search_range)
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

    def include_lost(self):
        """ Add source particles without any candidate to the subnets. """
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

    def merge_lost_subnets(self, search_range):
        """ Merge subnets that have lost features and that are closer than
        twice the search range together, in order to account for the possibility
        that relocated points will join subnets together."""
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

        source_hash = self.source_hash
        lost_coords = source_hash.predict(lost_source)
        dists, inds = source_hash.query(lost_coords, self.max_neighbors,
                                        rescale=True,
                                        search_range=search_range*2)
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
