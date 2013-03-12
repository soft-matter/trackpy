import trackpy.tracking as pt
import numpy as np
import pandas as pd

def track(features, search_range=5, memory=0, box_size=100):
    frames = []
    for frame_no, positions in features[['x', 'y']].groupby(features['frame']):
        frame = []
        frames.append(frame)
        for i, pos in positions.iterrows():
            frame.append(pt.PointND(frame_no, pos))
    
    hash_generator = lambda: pt.Hash_table((1000,1000), box_size)
    tracks = pt.link(frames, search_range, hash_generator, memory)
    probes = []
    for t in tracks:
        probe = pd.DataFrame(map(lambda x: x.pos, t.points), 
                             index=map(lambda x: x.t, t.points),
                             columns = ['x', 'y'])
        probes.append(probe)
    probes = pd.concat(probes, keys=np.arange(len(probes)), names=['probe', 'frame'])
    return probes.reset_index()

def bust_ghosts(tracks, threshold=100):
    b = t.groupby('probe').frame.transform(len) > threshold
    return tracks[b]
