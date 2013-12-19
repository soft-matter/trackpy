#Copyright 2012 Thomas A Caswell
#tcaswell@uchicago.edu
#http://jfi.uchicago.edu/~tcaswell
#
#This program is free software; you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation; either version 3 of the License, or (at
#your option) any later version.
#
#This program is distributed in the hope that it will be useful, but
#WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program; if not, see <http://www.gnu.org/licenses>.
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
from six.moves import zip

import numpy as np
from collections import deque


class Hash_table(object):
    '''
    :param dims: the range of the data to be put in the hash table.  0<data[k]<dims[k]
    :param box_size: how big each box should be in data units.  The same scale is used for all dimensions

    Basic hash table to fast look up of particles in the region of a given particle
    '''
    class Out_of_hash_excpt(Exception):
        """
        :py:exc:`Exception` for indicating that a particle is outside of the
        valid range for this hash table."""
        pass

    def __init__(self, dims, box_size):
        '''
        Sets up the hash table

        '''
        self.dims = dims                  # the dimensions of the data
        self.box_size = box_size          # the size of boxes to use in the units of the data
        self.hash_dims = np.ceil(np.array(dims) / box_size)

        self.hash_table = [[] for j in range(int(np.prod(self.hash_dims)))]
        self.spat_dims = len(dims)        # how many spatial dimensions
        self.cached_shifts = None
        self.cached_rrange = None
        self.strides = np.cumprod(np.concatenate(([1], self.hash_dims[1:])))[::-1]

    def get_region(self, point, rrange):
        '''
        :param point: point to find the features around
        :param rrange: the size of the ball to search in


        Returns all the particles with in the region of maximum radius
        rrange in data units


        can raise :py:exc:`Out_of_hash_excpt`
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
        :param point: object representing the feature to add to the hash table

        Adds the `point` to the hash table.  Assumes that :py:attr:`point.pos` exists and
        is the array-like.




        can raise :py:exc:`~Hash_table.Out_of_hash_excpt`

        """
        cord = np.floor(np.asarray(point.pos) / self.box_size)
        hash_size = self.hash_dims
        if any(cord >= hash_size) or any(cord < 0):
            raise Hash_table.Out_of_hash_excpt("cord out of range")
        indx = int(sum(cord * self.strides))
        self.hash_table[indx].append(point)


class Track(object):
    '''
    :param point: The first feature in the track if not  `None`.
    :type point: :py:class:`~trackpy.tracking.Point`

    Base class for objects to represent linked tracks.  Includes logic
    for adding, removing features to the track.  This can be sub-classed
    to provide additional track level computation as needed.


    '''
    count = 0

    def __init__(self, point=None):
        self.points = []
        # will take initiator point
        if not point is None:
            self.add_point(point)

        self.indx = Track.count           # unique id
        Track.count += 1

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
        :type point:  :py:class:`~trackpy.tracking.Point`

        Appends the point to this track. '''
        self.points.append(point)
        point.add_to_track(self)

    def remove_point(self, point):
        '''
        :param point: point to remove from this track
        :type point:  :py:class:`~trackpy.tracking.Point`

        removes a point from this track'''
        self.points.remove(point)
        point._track = None

    def last_point(self):
        '''
        :rtype: :py:class:`~trackpy.tracking.Point`

        Returns the last point on the track'''
        return self.points[-1]

    def __repr__(self):
        return "<%s %d>" % (self.__class__.__name__, self.indx)


class Point(object):
    '''
    Base class for point (features) used in tracking.  This class
    contains all of the general stuff for interacting with
    :py:class:`~trackpy.tracking.Track` objects.



    .. note:: To be used for tracking this class must be sub-classed to provide a :py:func:`distance` function.  Child classes **MUST** call :py:func:`Point.__init__`.  (See :py:class:`~trackpy.tracking.PointND` for example. )
    '''
    count = 0

    def __init__(self):
        self._track = None
        self.uuid = Point.count         # unique id for __hash__
        Point.count += 1

    ## def __eq__(self, other):
    ##     return self.uuid == other.uuid

    ## def __neq__(self, other):
    ##     return not self.__eq__(other)

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


class PointND(Point):
    '''
    :param t: a time-like variable.
    :param pos: position of feature
    :type pos: iterable of length d

    Version of :py:class:`Point` for tracking in flat space with
    non-periodic boundary conditions.
    '''

    def __init__(self, t, pos):
        Point.__init__(self)                  # initialize base class
        self.t = t                            # time
        self.pos = np.asarray(pos)            # position in ND space

    def distance(self, other_point):
        '''
        :param other_point: point to get distance to.
        :type other_point: :py:class:`~trackpy.tracking.Point`

        Returns the absolute distance between this point and other_point

        '''
        return np.sqrt(np.sum((self.pos - other_point.pos) ** 2))

    def __str__(self):
        return "({t}, {p})".format(t=self.t, p=self.pos)

    def __repr__(self):
        coords = '(' + (', '.join(["{:.3f}"]*len(self.pos))).format(*self.pos) + ')'
        track = " in Track %d" % self.track.indx if self.track else ""
        return "<%s at %d, " % (self.__class__.__name__, self.t) + coords + track + ">" 


def link_full(levels, dims, search_range, hash_cls, memory=0, track_cls=Track):
    '''
    :param levels: Nested iterables of :py:class:`~trackpy.tracking.Point` objects
    :type levels: Iterable of iterables
    :param dims: The dimensions of the data
    :param hash_cls: A class that provides :py:func:`add_particle` and
                :py:func:`get_region` and has a two argument constructor
    :param search_range: the maximum distance features can move between frames
    :param memory: the maximum number of frames that a feature can skip along a track
    :param track_cls: The class to use for the returned track objects
    :type track_cls: :py:class:`~trackpy.tracking.Track`

    Generic version of linking algorithm, should work for any
    dimension.  All information about dimensionality and the metric
    are encapsulated in the hash_table and
    :py:class:`~trackpy.tracking.Point` objects.

    .. deprecated:: 0.2
    '''
    hash_generator = lambda: hash_cls(dims, search_range)
    return link(levels, search_range, hash_generator, memory, track_cls)


def link(levels, search_range, hash_generator, memory=0, track_cls=Track):
    '''
    :param levels: Nested iterables of :py:class:`~trapy.tracking.Point` objects
    :type levels: Iterable of iterables
    :param search_range: the maximum distance features can move between frames
    :param hash_generator: a callable that will return a new empty object that supports :py:func:`add_particle` and
                :py:func:`get_region`
    :param memory: the maximum number of frames that a feature can skip along a track
    :param track_cls: The class to use for the returned track objects
    :type track_cls: :py:class:`~trackpy.tracking.Track`

    Generic version of linking algorithm, should work for any
    dimension.  All information about dimensionality and the metric
    are encapsulated in the hash_table and
    :py:class:`~trackpy.tracking.Point` objects.

    '''

    #    print "starting linking"
    # initial source set
    prev_set = set(levels[0])
    prev_hash = hash_generator()
    # set up the particles in the previous level for
    # linking
    for p in prev_set:
        prev_hash.add_point(p)
        p.forward_cands = []

    # assume everything in first level starts a track
    # initialize the master track list with the points in the first level
    track_lst = [track_cls(p) for p in prev_set]
    mem_set = set()
    # fill in first 'prev' hash

    # fill in memory list of sets
    mem_history = []
    for j in range(memory):
        mem_history.append(set())

    for cur_level in levels[1:]:
        # make a new hash object
        cur_hash = hash_generator()

        # create the set for the destination level
        cur_set = set(cur_level)
        # create a second copy that will be used as the source in
        # the next loop
        tmp_set = set(cur_level)

        # fill in first 'cur' hash and set up attributes for keeping
        # track of possible connections
        for p in cur_set:
            cur_hash.add_point(p)
            p.back_cands = []
            p.forward_cands = []
        # sort out what can go to what
        for p in cur_level:
            # get
            work_box = prev_hash.get_region(p, search_range)
            for wp in work_box:
                # this should get changed to deal with squared values
                # to save an eventually square root
                d = p.distance(wp)
                if d < search_range:
                    p.back_cands.append((wp, d))
                    wp.forward_cands.append((p, d))

        # sort the candidate lists by distance
        for p in cur_set:
            p.back_cands.sort(key=lambda x: x[1])
        for p in prev_set:
            p.forward_cands.sort(key=lambda x: x[1])

        new_mem_set = set()
        # while there are particles left to link, link
        while len(cur_set) > 0:
            p = cur_set.pop()
            bc_c = len(p.back_cands)
            # no backwards candidates
            if bc_c == 0:
                # add a new track
                track_lst.append(track_cls(p))
                # clean up tracking apparatus
                del p.back_cands
                # short circuit loop
                continue
            if bc_c == 1:
                # one backwards candidate
                b_c_p = p.back_cands[0]
                # and only one forward candidate
                b_c_p_0 = b_c_p[0]
                if len(b_c_p_0.forward_cands) == 1:
                    # add to the track of the candidate
                    b_c_p_0.track.add_point(p)
                    _maybe_remove(mem_set, b_c_p_0)
                    # clean up tracking apparatus
                    del p.back_cands
                    del b_c_p_0.forward_cands
                    prev_set.discard(b_c_p_0)
                    # short circuit loop
                    continue
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
                        prev_set.discard(c_sp[0])
                for sp in s_sn:
                    for c_dp in sp.forward_cands:
                        d_sn.add(c_dp[0])
                        cur_set.discard(c_dp[0])
                done_flg = (len(d_sn) == d_sn_sz) and (len(s_sn) == s_sn_sz)

            # add in penalty for not linking
            for _s in s_sn:
                _s.forward_cands.append((None, search_range))

            spl, dpl = _private_linker(s_sn, len(d_sn), search_range)

            # Identify the particles in the destination set that were not linked to
            d_remain = set(d for d in d_sn if d is not None)
            d_remain -= set(d for d in dpl if d is not None)
            for dp in d_remain:
                # if unclaimed destination particle, a track in born!
                track_lst.append(track_cls(dp))
                # clean up
                del dp.back_cands

            for sp, dp in zip(spl, dpl):
                # do linking and clean up
                if sp is not None and dp is not None:
                    sp.track.add_point(dp)
                    _maybe_remove(mem_set, sp)
                if dp is not None:
                    del dp.back_cands
                if sp is not None:
                    del sp.forward_cands
                    if dp is None:
                        # add the unmatched source particles to the new
                        # memory set
                        new_mem_set.add(sp)

        # Remember the source particles left unlinked that were not in
        # a subnetwork.
        for sp in prev_set:
            new_mem_set.add(sp)

        # set prev_hash to cur hash
        prev_hash = cur_hash
        if memory > 0:
            # identify the new memory points
            new_mem_set -= mem_set
            mem_history.append(new_mem_set)
            # remove points that are now too old
            mem_set -= mem_history.pop(0)
            # add the new points
            mem_set |= new_mem_set
            # add the memory particles to what will be the next source
            # set
            tmp_set |= mem_set
            # add memory points to prev_hash (to be used as the next source)
            for m in mem_set:
                # add points to the hash
                prev_hash.add_point(m)
                # re-create the forward_cands list
                m.forward_cands = []
        prev_set = tmp_set

        # add in the memory points
        # store the current level for use in next loop

    return track_lst


class SubnetOversizeException(Exception):
    '''An :py:exc:`Exception` to be raised when the sub-nets are too
    big to be efficiently linked.  If you get this then either reduce your search range
    or increase :py:attr:`sub_net_linker.MAX_SUB_NET_SIZE`'''
    pass


def recursive_linker_obj(s_sn, dest_size, search_range):
    snl = sub_net_linker(s_sn, dest_size, search_range)
    return zip(*snl.best_pairs)


class sub_net_linker(object):
    '''A helper class for implementing the Crocker-Grier tracking
    algorithm.  This class handles the recursion code for the sub-net linking'''
    MAX_SUB_NET_SIZE = 50

    def __init__(self, s_sn, dest_size, search_range):
        #        print 'made sub linker'
        self.s_sn = s_sn
        self.search_range = search_range
        self.s_lst = [s for s in s_sn]
        self.s_lst.sort(key=lambda x: len(x.forward_cands))
        self.MAX = len(self.s_lst)

        self.max_links = min(self.MAX, dest_size)
        self.best_pairs = None
        self.cur_pairs = deque([])
        self.best_sum = np.Inf
        self.d_taken = set()
        self.cur_sum = 0

        if self.MAX > sub_net_linker.MAX_SUB_NET_SIZE:
            raise SubnetOversizeException('sub net contains %d points' % self.MAX)
        # do the computation
        self.do_recur(0)

    def do_recur(self, j):
        cur_s = self.s_lst[j]
        for cur_d, dist in cur_s.forward_cands:
            tmp_sum = self.cur_sum + dist
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
                tmp_sum = self.cur_sum + self.search_range * (self.max_links - len(self.d_taken))
                if tmp_sum < self.best_sum:
                    self.best_sum = tmp_sum
                    self.best_pairs = list(self.cur_pairs)
            else:
                # re curse!
                self.do_recur(j + 1)
            # remove this step from the working
            self.cur_sum -= dist
            if cur_d is not None:
                self.d_taken.remove(cur_d)
            self.cur_pairs.pop()
        pass


def nonrecursive_link(source_list, dest_size, search_range):

    #    print 'non-recursive', len(source_list), dest_size
    source_list = list(source_list)
    source_list.sort(key=lambda x: len(x.forward_cands))
    MAX = len(source_list)

    max_links = min(MAX, dest_size)
    k_stack = deque([0])
    j = 0
    cur_back = deque([])
    cur_sum_stack = deque([0])

    best_sum = np.inf

    best_back = None
    cand_list_list = [c.forward_cands for c in source_list]
    cand_lens = [len(c) for c in cand_list_list]

    while j >= 0:
        # grab everything from the end of the stack
        cur_sum = cur_sum_stack[-1]
        if j >= MAX:
            # base case, no more source candidates,
            # save the current configuration if it's better than the current max
            # add penalty for not linking to particles in the destination set
            tmp_sum = cur_sum + search_range * (max_links - len([d for d in cur_back if d is not None]))
            if tmp_sum < best_sum:
                best_sum = cur_sum
                best_back = list(cur_back)

            j -= 1
            k_stack.pop()
            cur_sum_stack.pop()
            cur_back.pop()

            # print 'we have a winner'
            # print '-------------------------'
            continue

        # see if we have any forward candidates
        k = k_stack[-1]
        if k >= cand_lens[j]:
            # no more candidates to try, this branch is done
            j -= 1
            k_stack.pop()
            cur_sum_stack.pop()
            if j >= 0:
                cur_back.pop()
            # print 'out of cands'
            # print '-------------------------'
            continue

        # get the forward candidate
        cur_d, cur_dist = cand_list_list[j][k]

        tmp_sum = cur_sum + cur_dist
        if tmp_sum > best_sum:
            # nothing in this branch can do better than the current best
            j -= 1
            k_stack.pop()
            cur_sum_stack.pop()
            if j >= 0:
                cur_back.pop()
            # print 'total bail'
            # print '-------------------------'
            continue

        # advance the counter in the k_stack, the next time this level
        # of the frame stack is run the _next_ candidate will be run
        k_stack[-1] += 1
        # check if it's already linked
        if cur_d is not None and cur_d in cur_back:
            # this will run the loop with almost identical stack, but with advanced k
            # print 'already linked cur_d'
            # print '-------------------------'
            continue

        j += 1
        k_stack.append(0)
        cur_sum_stack.append(tmp_sum)
        cur_back.append(cur_d)

        # print '-------------------------'
    #    print 'done'
    return source_list, best_back

def _maybe_remove(s, p):
    # Begging forgiveness is faster than asking permission
    try:
        s.remove(p)
    except KeyError:
        pass

_private_linker = nonrecursive_link
