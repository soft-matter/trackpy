""" Functions to evaluate all possible links between two groups of features.

These are low-level functions that rigorously resolve cases when features
can be linked in more than one way.
"""

from collections import deque

import numpy as np

from .utils import SubnetOversizeException
from ..try_numba import try_numba_jit


def recursive_linker_obj(s_sn, dest_size, search_range, max_size=30, diag=False):
    snl = SubnetLinker(s_sn, dest_size, search_range, max_size=max_size)
    # In Python 3, we must convert to lists to return mutable collections.
    return [list(particles) for particles in zip(*snl.best_pairs)]


class SubnetLinker:
    """A helper class for implementing the Crocker-Grier tracking
    algorithm.  This class handles the recursion code for the sub-net linking"""
    def __init__(self, s_sn, dest_size, search_range, max_size=30):
        #        print 'made sub linker'
        self.s_sn = s_sn
        self.search_range = search_range
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
                if self.cur_sum < self.best_sum:
                    self.best_sum = self.cur_sum
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


def nonrecursive_link(source_list, dest_size, search_range, max_size=30, diag=False):
    source_list = list(source_list)
    source_list.sort(key=lambda x: len(x.forward_cands))
    MAX = len(source_list)

    if MAX > max_size:
        raise SubnetOversizeException("Subnetwork contains %d points" % MAX)

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
            if cur_sum < best_sum:
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

        tmp_sum = cur_sum + cur_dist**2
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


def numba_link(s_sn, dest_size, search_range, max_size=30, diag=False):
    """Recursively find the optimal bonds for a group of particles between 2 frames.

    This is only invoked when there is more than one possibility within
    ``search_range``.

    Note that ``dest_size`` is unused; it is determined from the contents of
    the source list.
    """
    # The basic idea: replace Point objects with integer indices into lists of Points.
    # Then the hard part runs quickly because it is just operating on arrays.
    # We can compile it with numba for outstanding performance.
    max_candidates = 9  # Max forward candidates we expect for any particle
    src_net = list(s_sn)
    nj = len(src_net) # j will index the source particles
    if nj > max_size:
        raise SubnetOversizeException('search_range (aka maxdisp) too large for reasonable performance '
                                      'on these data (sub net contains %d points)' % nj)
    # Build arrays of all destination (forward) candidates and their distances
    dcands = set()
    for p in src_net:
        dcands.update([cand for cand, dist in p.forward_cands])
    dcands = list(dcands)
    dcands_map = {cand: i for i, cand in enumerate(dcands)}
    # -1 is used by the numba code to signify a null link
    dcands_map[None] = -1
    # A source particle's actual candidates only take up the start of
    # each row of the array. All other elements represent the null link option
    # (i.e. particle lost)
    candsarray = np.ones((nj, max_candidates), dtype=np.int64) * -1
    distsarray = np.ones((nj, max_candidates), dtype=np.float64) * search_range
    ncands = np.zeros((nj,), dtype=np.int64)
    for j, sp in enumerate(src_net):
        ncands[j] = len(sp.forward_cands)
        if ncands[j] > max_candidates:
            raise SubnetOversizeException('search_range (aka maxdisp) too large for reasonable performance '
                                          'on these data (particle has %i forward candidates)' % ncands[j])
        candsarray[j,:ncands[j]] = [dcands_map[cand] for cand, dist in sp.forward_cands]
        distsarray[j,:ncands[j]] = [dist for cand, dist in sp.forward_cands]
        # Each source particle has a "null link" as its last forward_cand.
        # assert distsarray[j,ncands[j] - 1] == search_range
        # assert candsarray[j,ncands[j] - 1] == -1
    # The last column of distsarray should also be search_range,
    # so that the null link can be represented by "-1"
    # assert all(distsarray[:,-1] == search_range)

    # The assignments are persistent across levels of the recursion
    best_assignments = np.ones((nj,), dtype=np.int64) * -1
    cur_assignments = np.ones((nj,), dtype=np.int64) * -1
    tmp_assignments = np.zeros((nj,), dtype=np.int64)
    cur_sums = np.zeros((nj,), dtype=np.float64)
    # In the next line, distsarray is passed in quadrature so that adding distances works.
    loopcount = _numba_subnet_norecur(ncands, candsarray, distsarray**2, cur_assignments, cur_sums,
                                    tmp_assignments, best_assignments)
    if diag:
        for dr in dcands:
            try:
                dr.diag['subnet_iterations'] = loopcount
            except AttributeError:
                pass  # dr is "None" -- dropped particle
    source_results = list(src_net)
    dest_results = [dcands[i] if i >= 0 else None for i in best_assignments]
    return source_results, dest_results

@try_numba_jit(nopython=True)
def _numba_subnet_norecur(ncands, candsarray, dists2array, cur_assignments,
                          cur_sums, tmp_assignments, best_assignments):
    """Find the optimal track assignments for a subnetwork, without recursion.

    This is for nj source particles. All arguments are arrays with nj rows.

    cur_assignments, tmp_assignments are just temporary registers of length nj.
    best_assignments is modified in place.
    Returns the number of assignments tested (at all levels). This is basically
    proportional to time spent.
    """
    nj = candsarray.shape[0]
    tmp_sum = 0.
    best_sum = 1.0e23
    j = 0
    loopcount = 0  # Keep track of iterations. This should be an int64.
    while 1:
        loopcount += 1
        delta = 0 # What to do at the end
        # This is an endless loop. We go up and down levels of recursion,
        # and emulate the mechanics of nested "for" loops, using the
        # blocks of code marked "GO UP" and "GO DOWN". It's not pretty.

        # Load state from the "stack"
        i = tmp_assignments[j]
        #if j == 0:
        #    print i, j, best_sum
        #    sys.stdout.flush()
        if i >= ncands[j]:
            # We've exhausted possibilities at this level, including the
            # null link; make no more changes and go up a level
            #### GO UP
            delta = -1
        else:
            tmp_sum = cur_sums[j] + dists2array[j, i]
            if tmp_sum > best_sum:
                # if we are already greater than the best sum, bail. we
                # can bail all the way out of this branch because all
                # the other possible connections (including the null
                # connection) are more expensive than the current
                # connection, thus we can discard with out testing all
                # leaves down this branch
                #### GO UP
                delta = -1
            else:
                # We have to seriously consider this candidate.
                # We can have as many null links as we want, but the real particles are finite
                # This loop looks inefficient but it's what numba wants!
                flag = 0
                for jtmp in range(nj):
                    if cur_assignments[jtmp] == candsarray[j, i]:
                        if jtmp < j:
                            flag = 1
                if flag and candsarray[j, i] >= 0:
                    # We have already used this destination point and it is
                    # not null; try the next "i" instead
                    delta = 0
                else:
                    cur_assignments[j] = candsarray[j, i]
                    # OK, I guess we'll try this assignment
                    if j + 1 == nj:
                        # We have made assignments for all the particles,
                        # and we never exceeded the previous best_sum.
                        # This is our new optimum.
                        # print 'hit: %f' % best_sum
                        best_sum = tmp_sum
                        # This array is shared by all levels of recursion.
                        # If it's not touched again, it will be used once we
                        # get back to link_subnet
                        for jtmp in range(nj):
                            best_assignments[jtmp] = cur_assignments[jtmp]
                        #### GO UP
                        delta = -1
                    else:
                        # Try various assignments for the next particle
                        #### GO DOWN
                        delta = 1
        if delta == -1:
            if j > 0:
                j += -1
                tmp_assignments[j] += 1  # Try the next candidate at this higher level
                continue
            else:
                return loopcount
        elif delta == 1:
            j += 1
            cur_sums[j] = tmp_sum  # Floor for all subsequent sums
            tmp_assignments[j] = 0
        else:
            tmp_assignments[j] += 1


def drop_link(source_list, dest_size, search_range, max_size=30, diag=False):
    """Handle subnets by dropping particles.

    This is an alternate "link_strategy", selected by specifying 'drop',
    that simply refuses to solve the subnet. It ends the trajectories
    represented in source_list, and results in a new trajectory for
    each destination particle.

    One possible use is to quickly test whether a given search_range will
    result in a SubnetOversizeException."""
    if len(source_list) > max_size:
        raise SubnetOversizeException("Subnetwork contains %d points"
                                      % len(source_list))
    return [sp for sp in source_list], [None,] * len(source_list)


def subnet_linker_drop(source_set, dest_set, search_range, max_size=30,
                       **kwargs):
    """Handle subnets by dropping particles.

    This is an alternate "link_strategy", selected by specifying 'drop',
    that simply refuses to solve the subnet. It ends the trajectories
    represented in source_list, and results in a new trajectory for
    each destination particle.

    One possible use is to quickly test whether a given search_range will
    result in a SubnetOversizeException."""
    if len(source_set) == 0 and len(dest_set) == 1:
        # no backwards candidates: particle will get a new track
        return [None], [dest_set.pop()]
    elif len(source_set) == 1 and len(dest_set) == 1:
        # one backwards candidate and one forward candidate
        return [source_set.pop()], [dest_set.pop()]
    elif len(source_set) == 1 and len(dest_set) == 0:
        # particle is lost. Not possible with default Linker implementation.
        return [source_set.pop()], [None]

    if len(source_set) > max_size:
        raise SubnetOversizeException("Subnetwork contains %d points"
                                      % len(source_set))
    return [sp for sp in source_set] + [None] * len(dest_set), \
           [None] * len(source_set) + [dp for dp in dest_set]


def subnet_linker_recursive(source_set, dest_set, search_range, **kwargs):
    if len(source_set) == 0 and len(dest_set) == 1:
        # no backwards candidates: particle will get a new track
        return [None], [dest_set.pop()]
    elif len(source_set) == 1 and len(dest_set) == 1:
        # one backwards candidate and one forward candidate
        return [source_set.pop()], [dest_set.pop()]
    elif len(source_set) == 1 and len(dest_set) == 0:
        # particle is lost. Not possible with default Linker implementation.
        return [source_set.pop()], [None]

    # Add the null candidate that is required by the subnet linker.
    # Forward candidates were already sorted by Linker.assign_links()
    for _s in source_set:
        _s.forward_cands.append((None, search_range))

    snl = SubnetLinker(source_set, len(dest_set), search_range, **kwargs)
    sn_spl, sn_dpl = [list(particles) for particles in zip(*snl.best_pairs)]

    for dp in dest_set - set(sn_dpl):
        # Unclaimed destination particle in subnet
        sn_spl.append(None)
        sn_dpl.append(dp)

    return sn_spl, sn_dpl


def subnet_linker_nonrecursive(source_set, dest_set, search_range, **kwargs):
    if len(source_set) == 0 and len(dest_set) == 1:
        # no backwards candidates: particle will get a new track
        return [None], [dest_set.pop()]
    elif len(source_set) == 1 and len(dest_set) == 1:
        # one backwards candidate and one forward candidate
        return [source_set.pop()], [dest_set.pop()]
    elif len(source_set) == 1 and len(dest_set) == 0:
        # particle is lost. Not possible with default Linker implementation.
        return [source_set.pop()], [None]

    # Add the null candidate that is required by the subnet linker.
    # Forward candidates were already sorted by Linker.assign_links()
    for _s in source_set:
        _s.forward_cands.append((None, search_range))

    sn_spl, sn_dpl = nonrecursive_link(source_set, len(dest_set), search_range, **kwargs)

    for dp in dest_set - set(sn_dpl):
        # Unclaimed destination particle in subnet
        sn_spl.append(None)
        sn_dpl.append(dp)

    return sn_spl, sn_dpl

def subnet_linker_numba(source_set, dest_set, search_range,
                        hybrid=True, **kwargs):
    """Link a subnet using a numba-accelerated algorithm.

    Since this is meant to be the highest-performance option, it
    has some special behaviors:

    - Each source particle's forward_cands must be sorted by distance.
    - If the 'hybrid' option is true, subnets with only 1 source or
      destination particle, or with at most 4 source particles and
      4 destination particles, are solved using the recursive
      pure-Python algorithm, which has much less overhead since
      it does not convert to a numpy representation.
    """
    lss = len(source_set)
    lds = len(dest_set)
    if lss == 0 and lds == 1:
        # no backwards candidates: particle will get a new track
        return [None], [dest_set.pop()]
    elif lss == 1 and lds == 1:
        # one backwards candidate and one forward candidate
        return [source_set.pop()], [dest_set.pop()]
    elif lss == 1 and lds == 0:
        # particle is lost. Not possible with default Linker implementation.
        return [source_set.pop()], [None]

    # Add the null candidate that is required by the subnet linker.
    # Forward candidates were already sorted by Linker.assign_links()
    for _s in source_set:
        _s.forward_cands.append((None, search_range))

    # Shortcut for small subnets, because the numba linker has significant overhead
    if (lds == 1 or lss == 1 or (lds <= 3 and lss <= 3)) and hybrid:
        sn_spl, sn_dpl = recursive_linker_obj(source_set, lds, search_range, **kwargs)
    else:
        sn_spl, sn_dpl = numba_link(source_set, lds, search_range, **kwargs)

    for dp in dest_set - set(sn_dpl):
        # Unclaimed destination particle in subnet
        sn_spl.append(None)
        sn_dpl.append(dp)

    return sn_spl, sn_dpl
