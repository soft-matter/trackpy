import trackpy.tracking as pt
import numpy as np
import pandas as pd

class Feature(pt.PointND):
    "Extends pt.PointND to carry meta information from feature identification."
    def add_meta(self, mass, size, ecc, signal, ep):
        self.mass = mass
        self.size = size
        self.ecc = ecc
        self.signal = signal
        self.ep = ep

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
    for frame_no, fs in features.groupby(features['frame']):
        frame = []
        frames.append(frame)
        for i, vals in fs.iterrows():
            # Assume first two columns are x, y, and last is frame number.
            f = Feature(vals[-1], (vals[0], vals[1]))
            f.add_meta(*vals[2:-1])
            frame.append(f)
    
    hash_generator = lambda: pt.Hash_table((1300,1000), box_size)
    print "Doing the actual work..."
    tracks = pt.link(frames, search_range, hash_generator, memory)
    print "Organizing the output..."
    probes = []
    for t in tracks:
        probe = pd.DataFrame(
            map(lambda x: list(x.pos) + [x.mass, x.size, x.ecc, x.signal, x.ep], t.points), 
            index=map(lambda x: x.t, t.points),
            columns = ['x', 'y', 'mass', 'size', 'ecc', 'signal', 'ep'])
        probes.append(probe)
    probes = pd.concat(probes, keys=np.arange(len(probes)), 
                       names=['probe', 'frame'])
    return probes.reset_index()

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
