import trackpy.tracking as pt
import numpy as np
import pandas as pd

def track(features, search_range=5, memory=0, box_size=0.5):
    frames = []
    for frame_no, positions in features[['x', 'y']].groupby(features['frame']):
        frame = []
        frames.append(frame)
        for i, pos in positions.iterrows():
            frame.append(pt.PointND(frame_no, pos))
    
    hash_generator = lambda: pt.Hash_table((1000,1000), box_size)
    print 'I have a hash generator.'
    tracks = pt.link(frames, search_range, hash_generator, memory)
    probes = []
    for t in tracks:
        probe = pd.DataFrame(map(lambda x: [x.t, *x.pos], t.points), columns = [['frame', 'x', 'y']])
        probes.append(probe)
    return pd.concat(probes, keys=np.arange(len(probes)))
