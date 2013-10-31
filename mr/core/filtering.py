"""Simple functions that eliminate spurrious trajectories
by wrapping pandas group-by and filter capabilities."""


def filter_stubs(tracks, threshold=100):
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

def filter_clusters(tracks, quantile=0.8, threshold=None):
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

bust_ghosts = filter_stubs
bust_clusters = filter_clusters
