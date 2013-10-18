import itertools
import numpy as np
import pandas as pd
import trackpy.tracking as pt

class Feature(pt.PointND):
    "Extends pt.PointND to carry meta information from feature identification."
    def tag_id(self, id):
        self.id = id # unique ID derived from sequential index
                     # of features DataFrame

def track(features, search_range=5, memory=0, hash_size=None, box_size=None):
    """Link features into trajectories.

    Parameters
    ----------
    features : DataFrame including x, y, frame
    search_range : maximum displacement of a probe between two frames
        Default is 5 px.
    memory : Number of frames through which a probe is allowed to "disappear"
        and reappear and be considered the same probe. Default 0.
    hash_size : Dimensions of the search space, inferred from data by default.
    box_size : A parameter of the underlying algorith, defaults to same
        as search_range, which gives reasonably optimal performance.

    Note
    ----
    The index of the features DataFrame is dropped, and the result is given
    with a sequential integer index.
    """
    if box_size is None:
        box_size = search_range
    if hash_size is None:
        hash_size = np.ceil(features[['x', 'y']].max())
    print "Building Feature objects..."
    frames = []
    # Make a sequential index and promote it to a column called 'index'.
    trajectories = features.reset_index(drop=True).reset_index()
    for frame_no, fs in trajectories.groupby('frame'):
        frame = []
        frames.append(frame)
        for i, vals in fs.iterrows():
            f = Feature(vals['frame'], (vals['x'], vals['y']))
            f.tag_id(vals['index'])
            frame.append(f)
    del trajectories['index']
    
    hash_generator = lambda: pt.Hash_table(hash_size, box_size)
    print "Doing the actual work..."
    tracks = pt.link(frames, search_range, hash_generator, memory)
    print "Organizing the output..."
    trajectories['probe'] = np.nan
    for probe_id, t in enumerate(tracks):
        for p in t.points:
            trajectories.at[p.id, 'probe'] = probe_id
    return trajectories.sort(['probe', 'frame']).reset_index(drop=True)

def kdtree_track_dataframe(features, search_range):
    frames = (frame for frame in features.groupby('frame'))
    label_gen = kdtree_track(frames, search_range)
    features['probe'] = np.nan # placeholder
    while True:
        try:
            frame_no, labels = next(label_gen)
            features['probe'][features['frame'] == frame_no] = labels
        except StopIteration:
            break
    return features.sort(['probe', 'frame']).reset_index(drop=True)

def kdtree_track(frames, search_range, memory=0, position_cols=['x', 'y']):
    """Track particles.

    Parameters
    ----------
    frames : iterable, could be a generator for optimal memeory usage
    search_range : maximum displacement between frames
    memory : largest gap before a trajectory is considered ended
    position_cols : DataFrame column names (unlimited dimensions)
    """
    from scipy.spatial import KDTree
    from itertools import count
    max_disp = search_range # just a more succinct variable name
    
    # Process the first frame, and yield labels.
    frame_no, frame = next(frames)
    num_labels = len(frame[position_cols])
    labels = np.arange(num_labels)
    labeler = itertools.count(num_labels)
    trees = [KDTree(frame[position_cols])]
    prev_labels = labels
    yield frame_no, labels

    # Process the rest of the frames, yielding labels for each.
    for frame_no, frame in frames:
        # print 'frame_no', frame_no
        # Set up.
        trees.append(KDTree(frame[position_cols]))
        backward = _regularize(trees[-1].query_ball_tree(trees[-2], max_disp))
        forward = _regularize(trees[-2].query_ball_tree(trees[-1], max_disp))
        distances = trees[-1].sparse_distance_matrix(trees[-2], max_disp)
        labels = -1*np.ones(len(frame)) # an integer placeholder

        for i, b in enumerate(backward):
            # If it already has a label, skip it.
            if labels[i] != -1:
                continue 
            # To begin, two simple cases
            if len(b) == 0:
                # No backward candidates -- must be new.
                labels[i] = next(labeler)
                continue
            first_candidate = b[0] # maybe only candidate
            if len(b) == 1 and len(forward[first_candidate]) == 1:
                # One backward candidate which in turn
                # has one forward candidate -- link 'em!
                labels[i] = first_candidate
                continue

            # print 'after trivial:', labels
            # Initalize containers for a subnetwork of candidates.
            source = set()
            dest = set([i])
            stable = False
            # Fill the subnetwork iteratively until there are no more
            # connections in range.
            while not stable:
                subnet_size = map(len, [source, dest])
                [source.add(j) for d in dest for j in backward[d]]
                [dest.add(j) for s in source for j in forward[s]]
                stable = subnet_size == map(len, [source, dest])
            # print 'source, dest:', subnet_source, subnet_dest
            source = brute_force(distances, list(source), list(dest), max_disp)
            # Map source tree indices to existing labels using prev_labels.
            # print 'source_indexes:', source_indexes
            for d, s in zip(dest, source):
                if np.isnan(s):
                    labels[d] = next(labeler)
                else:
                    labels[d] = prev_labels[s]
                # print 'in labeling loop:', labels
        prev_labels = labels
        # print frame
        # print 'final:', frame_no, labels
        yield frame_no, labels

def _regularize(mixed_types):
    "Fix irregular output from KDTree, which gives one-elements lists as ints."
    return [[x] if not isinstance(x, list) else x for x in mixed_types]

def _distance(link, distances, penalty):
    """If link does not existing in the (sparse) matrix of distances
    return a penalty."""
    try:
        result = distances[link]
    except (IndexError, ValueError):
        return penalty
    else:
        if result > 0:
            return result
        else:  # result == 0
            return np.inf # particles not in range to link

def brute_force(distances, source, dest, search_range):
    "Return source tree index coresponding to each dest tree index."
    smallest_total = None
    count_missing = len(dest) - len(source)
    penalty = abs(count_missing)*search_range
    if count_missing == 0:
        pass
    elif count_missing > 0:
        source = np.append(source, [np.nan]*count_missing)
    else:  # count_missing < 0
        dest = np.append(dest, [np.nan]*-count_missing)
    distance = lambda link: _distance(link, distances, penalty)
    # print distances
    for source_permutation in itertools.permutations(source):
        proposed_links = [(s, d) for d, s in zip(dest, source_permutation)]
        total = np.sum([distance(link) for link in proposed_links])
        # print source_permutation, [distance(link) for link in proposed_links]
        if total < smallest_total or smallest_total is None:
            smallest_total = total 
            best_links = source_permutation
    return best_links
