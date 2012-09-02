from numpy import *

def interpolate(tracks):
    "Fill in gaps in a time series."
    steps = diff(tracks[0])
    for pos, step in enumerate(steps):
        if step > 1:
            # We have a gap.
            t0 = tracks[0][pos + ?]
            t1 = tracks[0][pos + ? + 1]
            # Get x0, x1, y0, y1, and compute slope and intercept.

def getdx(tracks, step):
    "I think this is just n-order differencing."
