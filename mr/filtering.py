"""Simple functions that eliminate spurrious trajectories
by wrapping pandas group-by and filter capabilities."""


def filter_stubs(tracks, threshold=100):
    """Filter out trajectories with few points. They are often specious.

    Parameters
    ----------
    tracks : DataFrame
        must include columns named 'frame' and 'probe'
    threshold : integer, default 100
        minimum number of points to survive

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
    tracks : DataFrame
        must include columns named 'probe' and 'size'
    quantile : number between 0 and 1
        quantile of probe 'size' above which to cut off
    threshold : number
        If specified, ignore quantile.

    Returns
    -------
    a subset of tracks
    """
    try:
        tracks['frame']
        tracks['probe']
    except KeyError:
        raise ValueError, "Tracks must contain columns 'frame' and 'probe'."
    if threshold is None:
        threshold = tracks['size'].quantile(quantile)

    f = lambda x: x['size'].mean() < threshold # filtering function
    grouped = tracks.reset_index(drop=True).groupby('probe')
    filtered = grouped.filter(f)
    return filtered.set_index('frame', drop=False)


def filter(tracks, condition_func):
    """A workaround for a bug in pandas 0.12

    Parameters
    ----------
    tracks : DataFrame
        must include column named 'probe'
    condition_func : function
        The function is applied to each group of data. It must
        return True or False.

    Returns
    -------
    DataFrame
        a subset of tracks
    """
    grouped = tracks.reset_index(drop=True).groupby('probe')
    filtered = grouped.filter(condition_func)
    return filtered.set_index('frame', drop=False)


bust_ghosts = filter_stubs
bust_clusters = filter_clusters
