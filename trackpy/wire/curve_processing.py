import numpy as np
import pandas as pd

def locate_jumps(data, z_window=10, f_window=7,
               maxima_window=100, significance=2):
    """Find the points when the B-field was stepped.

    Parameters
    ----------
    data : Series of angle data, indexed by frame or time
    z_window : Size of noise sample
        Default 10.
    f_window : Compare these points ahead and behind to detect a jump.
        Default 7.
    maxima_window : Minimum spacing of distinct jumps. Default 100.
    significance: Minimum significance of a jump; in units of sigma. Default 2.

    Returns
    -------
    array of positions where jumps occurred
    """
    # Each point's z-score in the context of the preceding ones.
    z = (data - pd.rolling_mean(data, z_window))/pd.rolling_std(data, z_window)
    # f = z_{before}^2 - z_{after}^2
    f = pd.rolling_sum(z.shift(-f_window)**2 - z**2, f_window)
    jumps = ((f == pd.rolling_max(f, maxima_window, center=True)) & \
             (f > 2*f_window*significance))
    print jumps.value_counts()
    return jumps[jumps].index.values
