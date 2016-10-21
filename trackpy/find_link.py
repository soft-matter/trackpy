from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
from six.moves import range
import warnings
import logging
import itertools
from collections import deque

import numpy as np
import pandas as pd
from scipy import ndimage

from .masks import (r_squared_mask, x_squared_masks, slice_image, mask_image,
                    slice_pad)
from .find import grey_dilation, drop_close
from .utils import (default_pos_columns, is_isotropic, cKDTree,
                    catch_keyboard_interrupt, validate_tuple)
from .preprocessing import bandpass
from .refine import center_of_mass

logger = logging.getLogger(__name__)


def query_point(pos1, pos2, max_dist):
    """ Return elements of pos2 that have at least one element of pos1 closer
    than max_dist """
    if is_isotropic(max_dist):
        if hasattr(max_dist, '__iter__'):
            max_dist = max_dist[0]
        kdtree = cKDTree(pos2, 30)
        found = kdtree.query_ball_point(pos1, max_dist)
    else:
        kdtree = cKDTree(pos2 / max_dist[np.newaxis, :], 30)
        found = kdtree.query_ball_point(pos1 / max_dist[np.newaxis, :], 1.)

    found = set([i for sl in found for i in sl])  # ravel
    if len(found) == 0:
        return
    else:
        return pos2[list(found)]


def characterize(coords, image, radius, isotropic=True, scale_factor=None):
    if scale_factor is None:
        try:
            scale_factor = image.metadata['scale_factor']
        except (AttributeError, KeyError):
            scale_factor = 1.
    ndim = len(radius)
    mass = np.empty(len(coords))
    signal = np.empty(len(coords))
    if isotropic:
        rg_mask = r_squared_mask(radius, ndim)  # memoized
        size = np.empty(len(coords))
    else:
        rg_mask = x_squared_masks(radius, ndim)  # memoized
        size_ax = tuple(range(1, ndim + 1))
        size = np.empty((len(coords), len(radius)))
    for i, coord in enumerate(coords):
        im, origin = slice_pad(image, coord, radius)
        im = mask_image(coord, im, radius, origin)
        _mass = np.sum(im)
        mass[i] = _mass
        signal[i] = np.max(im)

        if isotropic:
            size[i] = np.sqrt(np.sum(rg_mask * im) / _mass)
        else:
            size[i] = np.sqrt(ndim * np.sum(rg_mask * im,
                                            axis=size_ax) / _mass)

    result = dict(mass=mass / scale_factor, signal=signal / scale_factor)
    if isotropic:
        result['size'] = size
    else:
        for _size, key in zip(size.T, ['size_z', 'size_y', 'size_x'][-ndim:]):
            result[key] = _size
    return result


class SubnetOversizeException(Exception):
    pass


class TrackUnstored(object):
    @classmethod
    def set_counter(cls):
        cls.counter = itertools.count()

    def __init__(self, point=None):
        self.id = next(self.counter)
        if point is not None:
            self.add_point(point)

    def add_point(self, point):
        point.add_to_track(self)

    def incr_memory(self):
        try:
            self._remembered += 1
        except AttributeError:
            self._remembered = 1

    def report_memory(self):
        try:
            m = self._remembered
            del self._remembered
            return m
        except AttributeError:
            return 0

    def __repr__(self):
        return "<%s %d>" % (self.__class__.__name__, self.indx)


class PointND(object):
    @classmethod
    def set_counter(cls):
        cls.counter = itertools.count()

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
        self.back_cands = []
        self.forward_cands = []
        self.subnet = None
        self.relocate_neighbors = []

    def distance(self, other_point):
        return np.sqrt(np.sum((self.pos - other_point.pos) ** 2))

    def add_to_track(self, track):
        if self._track is not None:
            raise Exception("trying to add a particle already in a track")
        self._track = track

    def remove_from_track(self, track):
        if self._track != track:
            raise Exception("Point not associated with given track")
        track.remove_point(self)

    def in_track(self):
        return self._track is not None

    @property
    def track(self):
        return self._track


class TreeFinder(object):
    def __init__(self, points, search_range):
        """Takes a list of particles.
        """
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

    def rebuild(self):
        if len(self.points) == 0:
            self._kdtree = None
        else:
            coords = _get_pcoords(self.points) / self.search_range
            self._kdtree = cKDTree(coords, 15)
        # This could be tuned
        self._clean = True

    @property
    def kdtree(self):
        if not self._clean:
            self.rebuild()
        return self._kdtree

    @property
    def coords(self):
        if self._clean:
            if self._kdtree is None:
                return
            else:
                return self._kdtree.data * self.search_range
        else:
            return _get_pcoords(self.points)

    @property
    def coords_rescaled(self):
        if self._clean:
            if self._kdtree is None:
                return
            else:
                return self._kdtree.data
        else:
            return _get_pcoords(self.points) / self.search_range

    def to_dataframe(self):
        coords = self.coords
        if coords is None:
            return
        data = pd.DataFrame(coords, columns=default_pos_columns(self.ndim),
                            index=[p.uuid for p in self.points])
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


class Subnets(object):
    def __init__(self, source_hash, dest_hash, max_neighbors=10):
        self.max_neighbors = max_neighbors
        self.source_hash = source_hash
        self.dest_hash = dest_hash
        self.reset()
        self.add_candidates()

    def reset(self):
        self._subnets = dict()
        for p in self.source_hash.points:
            p.forward_cands = []
            p.subnet = None
        for i, p in enumerate(self.dest_hash.points):
            p.backward_cands = []
            p.subnet = i
            self._subnets[i] = set(), {p}

    def add_candidates(self, source_hash=None, dest_hash=None):
        if source_hash is None:
            source_hash = self.source_hash
        if dest_hash is None:
            dest_hash = self.dest_hash
        if len(source_hash.points) == 0 or len(dest_hash.points) == 0:
            return
        dists, inds = source_hash.kdtree.query(dest_hash.coords_rescaled,
                                               self.max_neighbors,
                                               distance_upper_bound=1+1e-7)
        nn = np.sum(np.isfinite(dists), 1)  # Number of neighbors of each particle
        for i, p in enumerate(dest_hash.points):
            for j in range(nn[i]):
                wp = source_hash.points[inds[i, j]]
                p.back_cands.append((wp, dists[i, j]))
                wp.forward_cands.append((p, dists[i, j]))
                self.update_subnets(wp, p)

    def sort_candidates(self):
        for p in self.source_hash.points:
            p.forward_cands.sort(key=lambda x: x[1])
        for p in self.dest_hash.points:
            p.back_cands.sort(key=lambda x: x[1])

    def add_dest_points(self, source_points, dest_points):
        """Dest_hash should only contain new points"""
        # TODO is kdtree really faster here than brute force ?
        if len(dest_points) == 0:
            return
        source_hash = TreeFinder(source_points, self.source_hash.search_range)
        dest_hash = TreeFinder(dest_points, self.dest_hash.search_range)
        dists, inds = source_hash.kdtree.query(dest_hash.coords_rescaled,
                                               max(len(source_points), 2),
                                               distance_upper_bound=1+1e-7)
        nn = np.sum(np.isfinite(dists), 1)  # Number of neighbors of each particle
        for i, dest in enumerate(dest_hash.points):
            for j in range(nn[i]):
                source = source_hash.points[inds[i, j]]
                dest.back_cands.append((source, dists[i, j]))
                source.forward_cands.append((dest, dists[i, j]))
                # source particle always has a subnet, add the dest particle
                self._subnets[source.subnet][1].add(dest)
                dest.subnet = source.subnet

        # sort candidates again because they might have changed
        for p in source_hash.points:
            p.forward_cands.sort(key=lambda x: x[1])
        for p in dest_hash.points:
            p.back_cands.sort(key=lambda x: x[1])

    def update_subnets(self, source, dest):
        i1 = source.subnet
        i2 = dest.subnet
        if i1 is None and i2 is None:
            raise ValueError("No subnet for added destination particle")
        if i1 == i2:  # if a and b are already in the same subnet, do nothing
            return
        if i1 is None:  # source did not belong to a subset before
            # just add it
            self._subnets[i2][0].add(source)
            source.subnet = i2
        elif i2 is None:  # dest did not belong to a subset before
            # just add it
            self._subnets[i1][1].add(dest)
            dest.subnet = i1
        else:   # source belongs to subset i1 before
            # merge the subnets
            self._subnets[i2][0].update(self._subnets[i1][0])
            self._subnets[i2][1].update(self._subnets[i1][1])
            # update the subnet identifiers per point
            for p in itertools.chain(*self._subnets[i1]):
                p.subnet = i2
            # and delete the old source subnet
            del self._subnets[i1]

    def merge_lost_subnets(self):
        """Merge subnets having lost particles, normally r = 2 * search range"""
        # add source particles without any destination particle for relocation
        if len(self._subnets) > 0:
            counter = itertools.count(start=max(self._subnets) + 1)
        else:
            counter = itertools.count()
        for p in self.source_hash.points:
            if len(p.forward_cands) == 0:
                subnet = next(counter)
                self._subnets[subnet] = {p}, set()
                p.subnet = subnet

        # list subnets that have lost particles
        lost_source = []
        for key in self._subnets:
            source, dest = self._subnets[key]
            shortage = len(source) - len(dest)
            if shortage > 0:
                lost_source.extend(source)

        if len(lost_source) == 0:
            return
        lost_hash = TreeFinder(lost_source, self.source_hash.search_range)
        dists, inds = self.source_hash.kdtree.query(lost_hash.coords_rescaled,
                                                    self.max_neighbors,
                                                    distance_upper_bound=2+1e-7)
        nn = np.sum(np.isfinite(dists), 1)  # Number of neighbors of each particle
        for i, p in enumerate(lost_hash.points):
            for j in range(nn[i]):
                wp = self.source_hash.points[inds[i, j]]
                i1, i2 = p.subnet, wp.subnet
                if i1 != i2:
                    if i2 > i1:
                        i1, i2 = i2, i1
                    self._subnets[i2][0].update(self._subnets[i1][0])
                    self._subnets[i2][1].update(self._subnets[i1][1])
                    # update the subnet identifiers per point
                    for p in itertools.chain(*self._subnets[i1]):
                        p.subnet = i2
                    # and delete the old source subnet
                    del self._subnets[i1]

    def __iter__(self):
        return (self._subnets[key] for key in self._subnets)

    def lost(self):
        return [p for p in self.source_hash.points if p.subnet is None]


def _sort_key_spl_dpl(x):
    if x[0] is not None:
        return list(x[0].pos)
    else:
        return list(x[1].pos)


def find_link(reader, search_range, separation, diameter=None, memory=0,
              minmass=0, noise_size=1, smoothing_size=None, threshold=None,
              percentile=64, before_link=None, after_link=None, refine=False):
    """Find and link features in an image sequence

    Parameters
    ----------
    reader : pims.FramesSequence
    search_range : number or tuple
        maximum displacement of features between subsequent frames
    separation : number or tuple
        minimum separation distance between features
    diameter : number or tuple, optional
        feature diameter, used for characterization only. Default: separation.
    memory : number, optional
        number of frames that features are allowed to dissappear. Experimental.
        Default 0.
    minmass : number, optional
        minimum integrated intensity (in masked image). Default 0.
    noise_size : number or tuple, optional
        Size of Gaussian kernel with whith the image is convoluted for noise
        reduction. Default 1.
    smoothing_size : number or tuple, optional
        Size of rolling average box for background subtraction.
        By default, equals separation. This may introduce bias when refined on
        the background subtracted image!
    threshold : number, optional
        Threshold value for image. Default None.
    percentile : number, optional
        The upper percentile of intensities in the image are considered as
        feature locations. Default 64.
    before_link : function, optional
        This function is executed after the initial find of each frame, but
        but before the linking and relocating.
        It should take the following arguments (or **kwargs):
            coords, reader, image, image_proc, diameter, separation,
            search_range, margin, minmass.
        And it should return coords. coords is an ndarray containing the
        initially found feature coordinates. image and reader are unprocessed.
        image_proc is the processed image. Default None.
    after_link : function, optional
        This function is executed after the find and link of each frame. It
        should not change the number of features.
        It should take the following arguments (or **kwargs):
            features, reader, image, image_proc, diameter, separation,
            search_range, margin, minmass.
        And it should return features. features is a DataFrame containing the
        feature coordinates and characterization.
        image and reader are unprocessed. image_proc is the processed image.
        Default None.
    refine : boolean, optional
        Convenience parameter to do center-of-mass refinement. Cannot be used
        combined with an after_link function. Default False.
    """
    shape = reader[0].shape
    ndim = len(shape)
    if smoothing_size is None:
        smoothing_size = separation
    smoothing_size = validate_tuple(smoothing_size, ndim)
    separation = validate_tuple(separation, ndim)
    if diameter is None:
        diameter = separation
    else:
        diameter = validate_tuple(diameter, ndim)

    if refine:
        if after_link is not None:
            raise ValueError("Refine cannot be used together with after_link.")
        pos_columns = default_pos_columns(ndim)
        refine_columns = pos_columns[::-1] + ['mass']
        radius = tuple([d // 2 for d in diameter])
        def after_link(image, features, **kwargs):
            coords = features[pos_columns].values
            if len(coords) == 0:
                return features
            # no separation filtering, because we use precise grey dilation
            coords = center_of_mass(image, image, radius, coords, separation=0,
                                    characterize=False)
            features[refine_columns] = coords
            return features

    features = []
    proc_func = lambda x: bandpass(x, noise_size, smoothing_size, threshold)
    generator = _find_link_iter(reader, search_range, separation, diameter,
                                memory, percentile, minmass, proc_func,
                                before_link, after_link)
    for frame_no, f_frame in catch_keyboard_interrupt(generator, logger=logger):
        if f_frame is None:
            n_traj = 0
        else:
            n_traj = len(f_frame)
        logger.info("Frame {0}: {1} trajectories present.".format(frame_no,
                                                                  n_traj))
        if n_traj == 0:
            continue
        features.append(f_frame)

    features = pd.concat(features, ignore_index=False)
    return features


def _build_level(coords, frame_no, extra_data=None):
    if extra_data is None:
        return [PointND(frame_no, pos) for pos in coords]
    else:
        return [PointND(frame_no, pos,
                        extra_data={key: extra_data[key][i]
                                    for key in extra_data})
                for i, pos in enumerate(coords)]


def _get_pcoords(level):
    return np.array([p.pos for p in level])


def recursive_linker_obj(s_sn, dest_size, max_size=30):
    snl = SubnetLinker(s_sn, dest_size, max_size)
    # In Python 3, we must convert to lists to return mutable collections.
    return [list(particles) for particles in zip(*snl.best_pairs)]


class SubnetLinker(object):
    """A helper class for implementing the Crocker-Grier tracking
    algorithm.  This class handles the recursion code for the sub-net linking"""
    def __init__(self, s_sn, dest_size, max_size=30):
        # add in penalty for not linking
        for _s in s_sn:
            if len(_s.forward_cands) == 0:
                _s.forward_cands = [(None, 1.)]
            elif _s.forward_cands[-1][0] is not None:
                _s.forward_cands.append((None, 1.))
        self.s_sn = s_sn
        self.search_range = 1.
        self.max_size = max_size
        self.s_lst = [s for s in s_sn]
        self.s_lst.sort(key=lambda x: len(x.forward_cands))
        self.MAX = len(self.s_lst)

        self.max_links = min(self.MAX, dest_size)
        self.best_pairs = None
        self.cur_pairs = deque([])
        self.best_sum = np.Inf
        self.d_taken = set()
        self.cur_sum = 0

        if self.MAX > self.max_size:
            raise SubnetOversizeException("Subnetwork contains %d points"
                                          % self.MAX)
        # do the computation
        self.do_recur(0)

    def do_recur(self, j):
        cur_s = self.s_lst[j]
        for cur_d, dist in cur_s.forward_cands:
            tmp_sum = self.cur_sum + dist**2
            if tmp_sum > self.best_sum:
                # if we are already greater than the best sum, bail we
                # can bail all the way out of this branch because all
                # the other possible connections (including the null
                # connection) are more expensive than the current
                # connection, thus we can discard with out testing all
                # leaves down this branch
                return
            if cur_d is not None and cur_d in self.d_taken:
                # we have already used this destination point, bail
                continue
            # add this pair to the running list
            self.cur_pairs.append((cur_s, cur_d))
            # add the destination point to the exclusion list
            if cur_d is not None:
                self.d_taken.add(cur_d)
            # update the current sum
            self.cur_sum = tmp_sum
            # buried base case
            # if we have hit the end of s_lst and made it this far, it
            # must be a better linking so save it.
            if j + 1 == self.MAX:
                tmp_sum = self.cur_sum + self.search_range**2 * (
                    self.max_links - len(self.d_taken))
                if tmp_sum < self.best_sum:
                    self.best_sum = tmp_sum
                    self.best_pairs = list(self.cur_pairs)
            else:
                # re curse!
                self.do_recur(j + 1)
            # remove this step from the working
            self.cur_sum -= dist**2
            if cur_d is not None:
                self.d_taken.remove(cur_d)
            self.cur_pairs.pop()
        pass


class Linker(object):
    """See link_iter() for a description of parameters."""
    # Largest subnet we will attempt to solve.
    MAX_SUB_NET_SIZE = 30
    # For adaptive search, subnet linking should fail much faster.
    MAX_SUB_NET_SIZE_ADAPTIVE = 15
    # Maximum number of candidates per particle
    MAX_NEIGHBORS = 10

    def __init__(self, search_range, memory=0):
        self.memory = memory
        self.track_cls = TrackUnstored
        self.subnet_linker = recursive_linker_obj
        self.max_subnet_size = self.MAX_SUB_NET_SIZE
        self.subnet_counter = 0  # Unique ID for each subnet
        self.ndim = len(search_range)
        self.search_range = np.array(search_range)
        self.hash = None
        self.mem_set = set()

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

        self.hash = TreeFinder(_build_level(coords, t, extra_data),
                               self.search_range)
        return prev_hash

    def init_level(self, coords, t, extra_data=None):
        PointND.set_counter()
        TrackUnstored.set_counter()
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

    def to_dataframe(self):
        return self.hash.to_dataframe()

    def set_dataframe(self, value):
        if len(value) != len(self.hash.points):
            raise ValueError("Number of features has changed")
        self.coords = value[default_pos_columns(self.ndim)].values

    @property
    def points(self):
        return self.hash.points

    def next_level(self, coords, t, extra_data=None):
        prev_hash = self.update_hash(coords, t, extra_data)

        self.subnets = Subnets(prev_hash, self.hash, self.MAX_NEIGHBORS)
        spl, dpl = self.assign_links(prev_hash.points, self.hash.points)
        self.apply_links(spl, dpl)

    def assign_links(self, prev_points, cur_points):
        spl, dpl = [], []
        for source_set, dest_set in self.subnets:
            # no backwards candidates
            if len(source_set) == 0 and len(dest_set) == 1:
                # particle will get a new track
                dpl.append(dest_set.pop())
                spl.append(None)
                continue  # do next dest_set particle
            elif len(source_set) == 1 and len(dest_set) == 1:
                # one backwards candidate and one forward candidate
                dpl.append(dest_set.pop())
                spl.append(source_set.pop())
                continue  # do next dest_set particle

            sn_spl, sn_dpl = self.subnet_linker(source_set, len(dest_set))

            for dp in dest_set - set(sn_dpl):
                # Unclaimed destination particle in subnet
                sn_spl.append(None)
                sn_dpl.append(dp)

            self.subnet_counter += 1

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

            # Clean up
            if dp is not None:
                del dp.back_cands
            if sp is not None:
                del sp.forward_cands

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


class FindLinker(Linker):
    """
    diameter : integer or tuple
        Size used in grey dilation: relocates maxima that are local within
        diameter / 2. Relocate is done on masked images.
    separation : float or tuple
        The minimum distance between features, used in relocate, in pixels
        will locate no features closer than ``separation`` to other features
    search_range : float or tuple
        The maximum distance features can move between frames, in pixels
        will locate no features further than ``search_range`` from initial pos
    """
    # Largest subnet we will attempt to solve.
    MAX_SUB_NET_SIZE = 30
    # For adaptive search, subnet linking should fail much faster.
    MAX_SUB_NET_SIZE_ADAPTIVE = 15
    # Largest number of relocated particles to consider (per subnet)
    MAX_RELOCATE_COORDS = 10

    def __init__(self, diameter, separation, search_range, memory=0,
                 minmass=0, percentile=64):
        super(FindLinker, self).__init__(search_range, memory=memory)
        self.ndim = len(diameter)
        self.isotropic = is_isotropic(diameter)
        self.radius = tuple([int(d // 2) for d in diameter])
        self.separation = separation
        self.minmass = minmass  # in masked image
        self.percentile = percentile

        # For grey dilation: find the largest box that fits inside the ellipse
        # given by separation
        self.dilation_size = tuple([int(2 * s / np.sqrt(self.ndim))
                                   for s in self.separation])
        # slice_radius: radius for relocate mask
        # search_range + feature radius + 1
        self.slice_radius = tuple([int(s + r + 1)
                                   for (s, r) in zip(self.search_range,
                                                     self.radius)])
        # background_radius: radius to make sure the already located features
        # do not fall inside slice radius
        bg_radius = [sl + r + 1 for (sl, r) in zip(self.slice_radius,
                                                   self.radius)]
        # The big feature hashtable is normed to search_range. For performance,
        # we do not rebuild this large hashtable. apply the norm here and take
        # the largest value.
        self.max_dist_in_slice = max([a / b for (a, b) in zip(bg_radius,
                                                              self.search_range)])
        self.double_search_range = [2 * sr for sr in self.search_range]

        self.threshold = (None, None)

    def next_level(self, coords, t, image, extra_data=None):
        self.image = image
        self.curr_t = t
        super(FindLinker, self).next_level(coords, t, extra_data)

    def relocate(self, source_points, n=1):
        candidates, extra_data = self.get_relocate_candidates(source_points)
        if candidates is None:
            return set()
        else:
            n = min(n, len(candidates))
            points = _build_level(candidates[:n], self.curr_t,
                                  extra_data=extra_data)
        return set(points)

    def percentile_threshold(self, percentile):
        frame_no, threshold = self.threshold
        if self.curr_t != frame_no:
            not_black = self.image[np.nonzero(self.image)]
            if len(not_black) == 0:
                threshold = None
            else:
                threshold = np.percentile(not_black, percentile)
            self.threshold = (self.curr_t, threshold)
        return threshold

    def get_relocate_candidates(self, source_points):
        pos = _get_pcoords(source_points)

        # slice region around cluster
        im_unmasked, origin = slice_image(pos, self.image, self.slice_radius)

        # return when there is no intensity left
        if im_unmasked.sum() == 0:
            return None, None
        # mask image so that only regions up to slice_radius are visible
        im_masked = mask_image(pos, im_unmasked, self.slice_radius, origin,
                               invert=False)
        # return when there is no intensity left
        if im_masked.sum() == 0:
            return None, None

        # mask coords that were already found ('background')
        background = self.hash.query_points(pos, self.max_dist_in_slice)
        if background is not None:
            im_masked = mask_image(background, im_masked, self.separation,
                                   origin, invert=True)

        threshold = self.percentile_threshold(self.percentile)
        if threshold is None:  # completely black image
            return None, None
        if np.all(im_masked < threshold):  # image entirely below threshold
            return None, None
        # The intersection of the image with its dilation gives local maxima.
        dilation = ndimage.grey_dilation(im_masked, self.dilation_size,
                                         mode='constant')
        maxima = (im_masked == dilation) & (im_masked > threshold)
        if np.sum(maxima) == 0:   # no maxima
            return None, None
        coords = np.vstack(np.where(maxima)).T

        # drop points that are further than search range from any initial point
        coords = query_point(pos - origin, coords, self.search_range)
        if coords is None:
            return None, None

        # drop dimmer points that are closer than separation to each other
        coords = drop_close(coords, self.separation,
                            [im_masked[tuple(c)] for c in coords])
        if coords is None:
            return None, None

        try:
            scale_factor = self.image.metadata['scale_factor']
        except (AttributeError, KeyError):
            scale_factor = 1.
        extra_data = characterize(coords, im_masked, self.radius,
                                  self.isotropic, scale_factor)

        mass = extra_data['mass']
        mask = np.argsort(mass)[::-1][:np.sum(mass >= self.minmass)]
        for key in extra_data:
            extra_data[key] = extra_data[key][mask]
        return coords[mask] + origin, extra_data

    def assign_links(self, prev_points, cur_points):
        self.subnets.merge_lost_subnets()
        self.subnets.sort_candidates()
        spl, dpl = [], []
        for source_set, dest_set in self.subnets:
            shortage = len(source_set) - len(dest_set)
            if shortage > 0:
                new_cands = self.relocate(source_set, shortage)
                self.subnets.add_dest_points(source_set, new_cands)
            else:
                new_cands = set()

            if len(source_set) == 0 and len(dest_set) == 1:
                # no backwards candidates: particle will get a new track
                dpl.append(dest_set.pop())
                spl.append(None)
            elif len(source_set) == 1 and len(dest_set) == 0:
                # no forward candidates
                spl.append(source_set.pop())
                dpl.append(None)
            elif len(source_set) == 1 and len(dest_set) == 1:
                # one backwards candidate and one forward candidate
                dpl.append(dest_set.pop())
                spl.append(source_set.pop())
                for p in new_cands:
                    self.hash.add_point(p)
            else:
                sn_spl, sn_dpl = self.subnet_linker(source_set, len(dest_set))
                sn_dpl_set = set(sn_dpl)
                # claimed new destination particles
                for p in new_cands & sn_dpl_set:
                    self.hash.add_point(p)
                # unclaimed old destination particles
                unclaimed = (dest_set - sn_dpl_set) - new_cands
                sn_spl.extend([None] * len(unclaimed))
                sn_dpl.extend(unclaimed)

                self.subnet_counter += 1

                spl.extend(sn_spl)
                dpl.extend(sn_dpl)

        return spl, dpl


def _find_link_iter(reader, search_range, separation, diameter=None, memory=0,
                    percentile=64, minmass=0, proc_func=None,
                    before_link=None, after_link=None):

    shape = reader[0].shape
    ndim = len(shape)

    search_range = validate_tuple(search_range, ndim)
    separation = validate_tuple(separation, ndim)
    isotropic = is_isotropic(diameter)
    if proc_func is None:
        proc_func = lambda x: x

    if diameter is None:
        diameter = separation
    else:
        diameter = validate_tuple(diameter, ndim)
    radius = tuple([int(d // 2) for d in diameter])
    # Define zone of exclusion at edges of image, avoiding
    #   - Features with incomplete image data ("radius")
    #   - Extended particles that cannot be explored during subpixel
    #       refinement ("separation")
    #   - Invalid output of the bandpass step ("smoothing_size")
    margin = tuple([max(diam // 2, sep // 2 - 1) for (diam, sep) in
                    zip(diameter, separation)])

    # Check whether the margins are not covering the complete image
    if np.any([s <= 2*m for (s, m) in zip(shape, margin)]):
        # Check whether the image looks suspiciously like a multichannel image.
        if np.any([s <= 4 for s in shape]) and (ndim > 2):
            raise ValueError('One of the image dimensions is very small. '
                             'Please make sure that you are not using an RGB '
                             'or other multichannel (color) image.')
        else:
            raise ValueError('The feature finding margins are larger than the '
                             'image shape. Please use smaller radius, '
                             'separation or smoothing_size.')

    linker = FindLinker(diameter, separation, search_range, memory, minmass,
                        percentile)

    reader_iter = iter(reader)
    image = next(reader_iter)
    image_proc = proc_func(image)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coords = grey_dilation(image_proc, separation, percentile, margin,
                               precise=True)
    if before_link is not None:
        coords = before_link(coords=coords, reader=reader, image=image,
                             image_proc=image_proc,
                             diameter=diameter, separation=separation,
                             search_range=search_range,
                             margin=margin, minmass=minmass)
    extra_data = characterize(coords, image, radius, isotropic)
    mask = extra_data['mass'] >= minmass
    coords = coords[mask]
    for key in extra_data:
        extra_data[key] = extra_data[key][mask]
    linker.init_level(coords, image.frame_no, extra_data)
    features = linker.to_dataframe()
    if after_link is not None and features is not None:
        features = after_link(features=features, reader=reader, image=image,
                              image_proc=image_proc,
                              diameter=diameter, separation=separation,
                              search_range=search_range, margin=margin,
                              minmass=minmass)
        linker.set_dataframe(features)  # for next iteration

    yield image.frame_no, features

    for image in reader_iter:
        image_proc = proc_func(image)
        coords = grey_dilation(image_proc, separation, percentile, margin,
                               precise=True)
        if before_link is not None:
            coords = before_link(coords=coords, reader=reader, image=image,
                                 image_proc=image_proc,
                                 diameter=diameter, separation=separation,
                                 search_range=search_range,
                                 margin=margin, minmass=minmass)
        extra_data = characterize(coords, image, radius, isotropic)
        mask = extra_data['mass'] >= minmass
        coords = coords[mask]
        for key in extra_data:
            extra_data[key] = extra_data[key][mask]
        linker.next_level(coords, image.frame_no, image_proc, extra_data)
        features = linker.to_dataframe()
        if after_link is not None and features is not None:
            features = after_link(features=features, reader=reader, image=image,
                                  image_proc=image_proc,
                                  diameter=diameter, separation=separation,
                                  search_range=search_range, margin=margin,
                                  minmass=minmass)
            linker.set_dataframe(features)  # for next iteration
        yield image.frame_no, features
