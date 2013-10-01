import trackpy.tracking as pt
import numpy as np
import pandas as pd

class Feature(pt.PointND):
    "Extends pt.PointND to carry meta information from feature identification."
    def tag_id(self, id):
        self.id = id # unique ID derived from sequential index
                     # of features DataFrame

def track(features, search_range=5, memory=0, box_size=100):
    """Link features into trajectories.

    Parameters
    ----------
    features : DataFrame including x, y, frame
    search_range : maximum displacement of a probe between two frames
        Default is 5 px.
    memory : Number of frames through which a probe is allowed to "disappear"
        and reappear and be considered the same probe. Default 0.
    box_size : A parameter of the underlying algorithm.

    Note
    ----
    The index of the features DataFrame is dropped, and the result is given
    with a sequential integer index.
    """
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
    
    hash_generator = lambda: pt.Hash_table((1300,1000), box_size)
    print "Doing the actual work..."
    tracks = pt.link(frames, search_range, hash_generator, memory)
    print "Organizing the output..."
    trajectories['probe'] = np.nan
    for probe_id, t in enumerate(tracks):
        for p in t.points:
            trajectories.at[p.id, 'probe'] = probe_id
    return trajectories.sort(['probe', 'frame']).reset_index(drop=True)

def kdtree_track(f, max_disp):
    from scipy.spatial import KDTree
    from itertools import count
    position_cols, frame_col = ['x', 'y'], 'frame' # DataFrame column names; could generalize to 3D
    first_iteration = True
    t = f.copy() # TODO BAD MEMORY USE -- CHANGE LATER?
    t['probe'] = np.nan
    for frame_no, frame in f.groupby(frame_col):
        if first_iteration:
            trees = [KDTree(frame[position_cols])]
            probes = np.arange(len(frame[position_cols])) # give probes id numbers...
            t['probe'][f[frame_col] == frame_no] = probes
            c = count(len(frame[position_cols])) # ...and use this counter for any new probes
            def probe_id(i=None):
                if i is None:
                    return next(c)
            first_iteration = False
            continue

        # Set up.
        trees.append(KDTree(frame[position_cols]))
        backward = trees[-1].query_ball_tree(trees[-2], max_disp)
        forward = trees[-2].query_ball_tree(trees[-1], max_disp)
        distances = trees[-1].count_neighbors(trees[-2], max_disp)
        probes = -1*np.ones(len(frame)) # placeholder

        # Process probes with only one candidate in range.
        for i, b in enumerate(backward):
            if len(b) == 0:
                # no backward candidates
                probes[i] = probe_id()
            if len(b) == 1:
                # one backward candidate
                candidate = b[0]
                if len(forward[candidate]) == 1:
                    # unambiguous
                    probe[i] = candidate
                else:
                    # ambiguous
             
        distances = trees[-1].sparse_distance_matrix(trees[-2], max_disp).toarray()
        count_forward = (distances != 0)
        one_match = num_matches == 1 # boolean mask
        # Is the match a UNIQUE match?
        distances[distances == 0] = max_disp + 1 # Replace placeholder before using argmin.
        probes[one_match] = distances[one_match].argmin(1)
        mult_matches = num_matches > 1
        # TODO The following is obviously problematic.
        probes[mult_matches] = distances[mult_matches].argmin(1) 
        t['probe'][f[frame_col] == frame_no] = probes
    return t.sort(['probe', frame_col]).reset_index(drop=True)

def bust_ghosts(tracks, threshold=100):
    """Filter out trajectories with few points. They are often specious.

    Parameters
    ----------
    tracks : DataFrame with a 'probe' column
    threshold : minimum number of points to survive. 100 by default.

    Returns
    -------
    a subset of tracks
    """
    try:
        tracks['frame']
        tracks['probe']
    except KeyError:
        raise ValueError, "Tracks must contain columns 'frame' and 'probe'."
    grouped = tracks.reset_index(drop=True).groupby('probe')
    filtered = grouped.filter(lambda x: x.frame.count() >= threshold)
    return filtered.set_index('frame', drop=False)

def bust_clusters(tracks, quantile=0.8, threshold=None):
    """Filter out trajectories with a mean probe size above a given quantile.

    Parameters
    ----------
    tracks: DataFrame with 'probe' and 'size' columns
    quantile : quantile of probe 'size' above which to cut off
    threshold : If specified, ignore quantile.

    Returns
    -------
    a subset of tracks
    """
    if threshold is None:
        threshold = tracks['size'].quantile(quantile)
    f = lambda x: x['size'].mean() < threshold # filtering function
    return tracks.groupby('probe').filter(f)
