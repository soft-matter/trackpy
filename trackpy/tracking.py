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
from __future__ import division

import numpy as np
import scipy

class Hash_table(object):
    '''
    Basic hash table to fast look up of 
    '''
    class Out_of_hash_excpt(Exception):
        pass
    def __init__(self,dims,box_size):
        self.dims = dims                  # the dimensions of the data
        self.box_size = box_size          # the size of boxes to use in the units of the data
        self.hash_dims = np.ceil(np.array(dims)/box_size)

        self.hash_table = [[] for j in range(int(np.prod(self.hash_dims)))]
        self.spat_dims = len(dims)        # how many spatial dimensions
        self.cached_shifts = None
        self.cached_rrange = None
    def get_region(self,point, rrange):
        '''
        Returns all the particles with in the region of minimum radius
        rrange in data units
        '''
        
        center = np.floor(point.pos/self.box_size)

        rrange = int(np.ceil(rrange/ self.box_size))

        # check if we have already computed the shifts
        if rrange == self.cached_rrange and self.cached_shifts is not None:
            shifts = self.cached_shifts   # if we have, use them
        # Other wise, generate them
        else:
            if self.spat_dims == 2:
                shifts = [np.array([j,k]) 
                            for j in range(-rrange,rrange + 1) 
                            for k in range(-rrange,rrange + 1)]
            elif self.spat_dims ==3:
                shifts = [np.array([j,k,m])
                            for j in range(-rrange,rrange + 1)
                            for k in range(-rrange,rrange + 1)
                            for m in range(-rrange,rrange + 1)]
            else:
                raise NotImplementedError('only 2 and 3 dimensions implemented')
            self.cached_rrange = rrange   # and save them
            self.cached_shifts = shifts
        region = []
        for s in shifts:
            try:
                region.extend(self.hash_table[self.cord_to_indx(center + s)])
            except Hash_table.Out_of_hash_excpt:
                pass
        return region
    
    def add_point(self,point):
        self.hash_table[self.hash_fun(point.pos)].append(point)

    def hash_fun(self,cord):
        return self.cord_to_indx(np.floor(np.asarray(cord)/self.box_size))

    def cord_to_indx(self,cord):
        """Converts coordinate position to an index.  """
        assert len(cord) == len(self.hash_dims)
        hash_size = self.hash_dims
        cord = np.asarray(cord)
        if any(cord  >= hash_size) or any(cord < 0):
            raise Hash_table.Out_of_hash_excpt("cord out of range")
        indx = int(sum(cord * np.cumprod(np.concatenate(([1],hash_size[1:])))))

        return indx

    
class Track(object):
    '''
    Object to represent linked tracks.  Includes logic for adding, removing 
    '''
    count = 0
    def __init__(self,point=None):
        self.points = []
        # will take initiator point
        if not point is None:
            self.add_point(point)
                                        
        self.indx = Track.count           #unique id
        Track.count +=1
    def __iter__(self):
        return self.points.__iter__()
    def __len__(self):
        return len(self.points)

    def __eq__(self,other):
        return self.index == other.index

    def __neq__(self,other):
        return not self.__eq__(other)
    __hash__ = None
        
    def add_point(self,point):
        '''Adds a point to this track '''
        self.points.append(point)
        point.add_to_track(self)
    def remove_point(self,point):
        '''removes a point from this track'''
        self.points.remove(point)
        point.track = None
    def last_point(self):
        '''Returns the last point on the track'''
        return self.points[-1]
    def merge_track(self,to_merge_track):
        '''Merges the track add_to_track into the current track.
        Progressively moves points from the other track to this one.
        '''
    
        while len(to_merge_track.points) >0:
            cur_pt = to_merge_track.points.pop()
            if cur_pt.track != self:
                raise Exception
            cur_pt.track = None
            self.add_point(cur_pt)

class Point(object):
    '''
    Base class for point objects used in tracking.  This class contains all of the
    general stuff for interacting with tracks.

    Child classes **MUST** call Point.__init__
    '''
    count = 0
    def __init__(self):
        self.track = None
        self.uuid = Point.count         # unique id for __hash__
        Point.count +=1

    def __hash__(self):
        return self.uuid
    def __eq__(self,other):
        return self.uuid == other.uuid
    def __neq__(self,other):
        return not self.__eq__(other)
    
        
    def add_to_track(self,track):
        '''Adds a track to the point '''
        if self.track is not None:
            raise Exception("trying to add a particle already in a track")
        self.track = track
        
    def remove_from_track(self,track):
        '''Removes a point from the given track, error if not really
        in that track'''
        if self.track != track:
            raise Exception
        track.remove_point(self)

    

    def in_track(self):
        '''Returns if a point is in a track '''
        return  self.track is not None
    
class PointND(Point):
    '''
    Version of :py:class:`Point` for tracking 

    :py:attr:`Point1D_circ.q` is the parameter for the curve where the point is (maps to time in standard tracking)
    :py:attr:`Point1D_circ.phi` is the angle of the point along the parametric curve
    :py:attr:`Point1D_circ.v` any extra values that the point should carry
    '''

    
    def __init__(self,t,pos):
        Point.__init__(self)                  # initialize base class
        self.t = t                            # time
        self.pos = np.asarray(pos)            # position in ND space 

    def distance(self,point):
        '''
        Returns the absolute distance between two points
        '''
        return np.sqrt(np.sum((self.pos - point.pos)**2))
    

def link_full(levels,dims,search_range,hash_cls,memory=0,track_cls = Track ):
    '''    Does proper linking, dealing with the forward and backward
    networks.  This should work with any dimension, so long and the
    hash object and the point objects behave as expected.

    levels is a list of lists of points.  The linking in done between
    the inner lists.

    Expect hash_cls to have constructor that takes a single value and
    support add_particle and get_region

    expect particles to know what track they are in (p.track -> track)
    and know how far apart they are from another particle (p.distance(p2)
    returns absolute distance)

    dims is the dimension of the data in data units
    '''
    #    print "starting linking"
    # initial source set    
    prev_set  = set(levels[0])
    prev_hash = hash_cls(dims,search_range)
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
        cur_hash = hash_cls(dims,search_range)

        # create the set for the destination level
        cur_set = set(cur_level)
        # create a second copy that will be used as the source in
        # the next loop
        tmp_set = set(cur_level) 
        # memory set
        new_mem_set = set()
        
        # fill in first 'cur' hash and set up attributes for keeping
        # track of possible connections

        for p in cur_set:
            cur_hash.add_point(p)
            p.back_cands = []
            p.forward_cands = []
        # sort out what can go to what
        for p in cur_level:
            # get 
            work_box = prev_hash.get_region(p,search_range)
            for wp in work_box:
                # this should get changed to deal with squared values
                # to save an eventually square root
                d = p.distance(wp)
                if d< search_range:
                    p.back_cands.append((wp,d))
                    wp.forward_cands.append((p,d))


        # sort the candidate lists by distance
        for p in cur_set: p.back_cands.sort(key=lambda x: x[1])
        for p in prev_set: p.forward_cands.sort(key=lambda x: x[1])
        # while there are particles left to link, link
        while len(prev_set) > 0 and len(cur_set) > 0:
            p = cur_set.pop()
            bc_c = len(p.back_cands)
            # no backwards candidates
            if bc_c ==  0:
                # add a new track
                track_lst.append(track_cls(p))
                # clean up tracking apparatus 
                del p.back_cands
                # short circuit loop
                continue
            if bc_c ==1:
                # one backwards candidate
                b_c_p = p.back_cands[0]
                # and only one forward candidate
                if len(b_c_p[0].forward_cands) ==1:
                    # add to the track of the candidate
                    b_c_p[0].track.add_point(p)
                    # clean up tracking apparatus
                    del p.back_cands
                    del b_c_p[0].forward_cands
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

            snl = sub_net_linker(s_sn,search_range)
            

            spl,dpl = zip(*snl.best_pairs)
            # strip the distance information off the subnet sets and
            # remove the linked particles
            d_remain = set([d for d in d_sn])
            d_remain -= set(dpl)
            s_remain = set([s for s in s_sn])
            s_remain -= set(spl)

            for sp,dp in snl.best_pairs:
                # do linking and clean up
                sp.track.add_point(dp)
                del dp.back_cands
                del sp.forward_cands
            for sp in s_remain:
                # clean up
                del sp.forward_cands
            for dp in d_remain:
                # if unclaimed destination particle, a track in born!
                track_lst.append(track_cls(dp))
                # clean up
                del dp.back_cands
            # tack all of the unmatched source particles into the new
            # memory set
            new_mem_set |= s_remain

        # set prev_hash to cur hash
        prev_hash = cur_hash
        if memory > 0:
            # identify the new memory points
            new_mem_set -= mem_set
            mem_history.append(new_mem_set)
            # remove points that are now too old
            mem_set -= mem_history.pop(0)
            # add the new points
            mem_set |=new_mem_set
            # add the memory particles to what will be the next source
            # set
            tmp_set |=mem_set
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
    pass

class sub_net_linker(object):
    '''A helper class for implementing the Crocker-Grier tracking
    algorithm.  This class handles the recursion code for the sub-net linking'''
    MAX_SUB_NET_SIZE = 50
    
    def __init__(self,s_sn,search_range):
        self.s_sn = s_sn
        self.s_lst = [s for s in s_sn]
        self.MAX = len(self.s_lst)
        self.sr = search_range
        self.best_pairs = []
        self.cur_pairs = []
        self.best_sum = np.Inf
        self.d_taken = set()
        self.cur_sum = 0

        if self.MAX > sub_net_linker.MAX_SUB_NET_SIZE:
            raise SubnetOversizeException('sub net contains %d points'%self.MAX)
        # do the computation
        self.do_recur(0)
    def do_recur(self,j):
        cur_s = self.s_lst[j]
        for cur_d,dist in cur_s.forward_cands:
            tmp_sum = self.cur_sum + dist
            if tmp_sum > self.best_sum:
                # if we are already greater than the best sum, bail we
                # can bail all the way out of this branch because all
                # the other possible connections (including the null
                # connection) are more expensive than the current
                # connection, thus we can discard with out testing all
                # leaves down this branch
                return
            if cur_d in self.d_taken:
                # we have already used this destination point, bail
                continue
            # add this pair to the running list 
            self.cur_pairs.append((cur_s,cur_d))
            # add the destination point to the exclusion list 
            self.d_taken.add(cur_d)
            # update the current sum
            self.cur_sum = tmp_sum
            # buried base case
            # if we have hit the end of s_lst and made it this far, it
            # must be a better linking so save it.             
            if j +1 == self.MAX:
                self.best_sum = tmp_sum
                self.best_pairs = list(self.cur_pairs)
            else:
                # recurse!
                self.do_recur(j+1)
            # remove this step from the working 
            self.cur_sum -= dist
            self.d_taken.remove(cur_d)
            self.cur_pairs.pop()
        # try null link
        tmp_sum = self.cur_sum + self.sr
        if tmp_sum < self.best_sum:
            # add displacement penalty
            self.cur_sum = tmp_sum
            # buried base case
            if j +1 == self.MAX:
                self.best_sum = tmp_sum
                self.best_pairs = list(self.cur_pairs)
            else:
                # recurse!
                self.do_recur(j+1)
            # remove penalty 
            self.cur_sum-=self.sr
        pass
    


