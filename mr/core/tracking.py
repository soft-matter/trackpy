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
    return trajectories

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
