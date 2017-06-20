from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
from six.moves import range
import itertools

import numpy as np


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
