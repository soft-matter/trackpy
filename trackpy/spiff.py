import numpy as np
from pandas import DataFrame
import warnings

# The minimum number of features required to apply SPIFF correction.
# This is a conservative requirement (likely higher than it needs to be) and is
# subject to further optimization.
MIN_FEATURES = 50


def _spiff_correct_column(x):
    """Flatten the sub-pixel histogram of a 1-D position array via its CDF."""
    n = len(x)
    spiff = x % 1
    # Mirror around the pixel center and sort for an empirical CDF lookup.
    spiff_sorted = np.sort(np.where(spiff < 0.5, spiff, 1 - spiff))
    low = np.searchsorted(spiff_sorted, spiff) / n / 2
    high = 1 - np.searchsorted(spiff_sorted, 1 - spiff) / n / 2
    return np.floor(x) + np.where(spiff < 0.5, low, high)


def _merged_bins(labels, min_count):
    """Group ordinal `labels` into contiguous bins each with >= min_count members.

    Adjacent classes are merged (in ascending order) until a bin is large enough
    for a stable CDF; a small trailing remainder folds into the previous bin.
    Returns a list of boolean masks over `labels`.
    """
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)  # ascending
    groups, current, running = [], [], 0
    for value, count in zip(uniq, counts):
        current.append(value)
        running += count
        if running >= min_count:
            groups.append(current)
            current, running = [], 0
    if current:  # trailing remainder below the threshold
        if groups:
            groups[-1].extend(current)
        else:
            groups.append(current)
    return [np.isin(labels, group) for group in groups]


def apply_spiff(f: DataFrame, pos_columns=None,
                warn_if_insufficient=True, groupby='auto') -> DataFrame:
    """
    Removes pixel bias in a given list of features (using a single-pixel interior filling function),
    thereby improving sub-pixel accuracy.

    Parameters
    ----------
    f : DataFrame
        Features as returned by ``trackpy.locate`` or ``trackpy.batch``.
    pos_columns : list of column names, optional
        The position columns to correct. If None, defaults to ``['x', 'y']``
        (and ``'z'`` if present in ``f``).
    warn_if_insufficient : boolean, optional
        If True (default), emit a warning when there are fewer than
        ``MIN_FEATURES`` rows to build a correction from and the correction is
        skipped. Set to False for silent skipping (e.g. when called
        automatically via the ``spiff='auto'`` option of ``locate`` or
        ``batch``).
    groupby : column name, None, or 'auto', optional
        Correct each size class separately, using the named column as the class
        label. Polydisperse features have different pixel-locking signatures per
        size, so a single pooled correction would mix them. ``'auto'`` (default)
        uses the ``'diameter'`` column when present (as emitted by polydisperse
        ``locate``/``batch``) and otherwise pools all features. Pass None to
        force a single pooled correction. Adjacent classes are merged until each
        bin has ``MIN_FEATURES`` members.

    Returns
    -------
    DataFrame
        A copy of ``f`` with corrected positions, or ``f`` unchanged where
        there are too few features.

    Notes
    -----
    The algorithm used is inspired by "Analysis and correction of errors in
    nanoscale particle tracking using the Single-pixel interior filling function
    (SPIFF) algorithm" paper (see below).
    The accuracy of this algorithm improves with the number of features. When
    tracking features across multiple frames (e.g. in a video), consider locating
    the features across all frames first (using tp.batch) before applying this function
    (as opposed to applying this function for each individual frame).
    If f contains fewer than ``MIN_FEATURES`` features, f is returned as-is,
    due to lack of data.

    Citations
    -----
    Yifat, Y., Sule, N., Lin, Y. et al.
    Analysis and correction of errors in nanoscale particle tracking using the
    Single-pixel interior filling function (SPIFF) algorithm. Sci Rep 7, 16553 (2017).
    https://doi.org/10.1038/s41598-017-14166-6
    """
    if pos_columns is None:
        if 'z' in f:
            pos_columns = ['x', 'y', 'z']
        else:
            pos_columns = ['x', 'y']

    if groupby == 'auto':
        groupby = 'diameter' if 'diameter' in f else None

    # A single pooled correction assumes one bias signature for all features.
    if groupby is None:
        if len(f) < MIN_FEATURES:
            if warn_if_insufficient:
                warnings.warn(
                    "Not enough features ({n} < {min_n}) to apply SPIFF "
                    "sub-pixel bias correction; returning features unchanged. "
                    "Consider running on a larger batch of frames.".format(
                        n=len(f), min_n=MIN_FEATURES))
            return f
        f = f.copy()
        for col in pos_columns:
            f[col] = _spiff_correct_column(np.asarray(f[col]))
        return f

    # Size-class-aware: correct within bins of the (ordinal) grouping column, so
    # distinct per-size bias signatures are not mixed into one pooled histogram.
    f = f.copy()
    corrected_any = False
    for mask in _merged_bins(f[groupby].values, MIN_FEATURES):
        if mask.sum() < MIN_FEATURES:
            continue
        index = f.index[mask]
        for col in pos_columns:
            f.loc[index, col] = _spiff_correct_column(np.asarray(f.loc[index, col]))
        corrected_any = True
    if warn_if_insufficient and not corrected_any:
        warnings.warn(
            "Not enough features ({n} < {min_n}) to apply SPIFF sub-pixel bias "
            "correction; returning features unchanged. Consider running on a "
            "larger batch of frames.".format(n=len(f), min_n=MIN_FEATURES))
    return f
