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
    """
    print "Building Feature objects..."
    frames = []
    # Make a sequential index and promote it to a column called 'index'.
    trajectories = features.reset_index(drop=True).reset_index()
    print trajectories.columns
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
    """Discard trajectories with few points. They are often specious.

    Parameters
    ----------
    tracks : DataFrame of with a 'probe' column
    threshold : minimum number of points to survive. 100 by default.

    Returns
    -------
    tracks, culled
    """
    b = tracks.groupby('probe').frame.transform(len) > threshold
    return tracks[b]
