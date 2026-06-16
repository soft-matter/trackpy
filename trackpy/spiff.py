import numpy as np
from pandas import DataFrame
import warnings

# The minimum number of features required to apply SPIFF correction.
# This is a conservative requirement (likely higher than it needs to be) and is
# subject to further optimization.
MIN_FEATURES = 50


def apply_spiff(f: DataFrame, pos_columns=None,
                warn_if_insufficient=True) -> DataFrame:
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
        If True (default), emit a warning when ``f`` contains fewer than
        ``MIN_FEATURES`` rows and the correction is skipped. Set to False
        for silent skipping (e.g. when called automatically via the
        ``spiff='auto'`` option of ``locate`` or ``batch``).

    Returns
    -------
    DataFrame
        A copy of ``f`` with corrected positions, or ``f`` unchanged if
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
    if len(f) < MIN_FEATURES:
        if warn_if_insufficient:
            warnings.warn(
                "Not enough features ({n} < {min_n}) to apply SPIFF "
                "sub-pixel bias correction; returning features unchanged. "
                "Consider running on a larger batch of frames.".format(
                    n=len(f), min_n=MIN_FEATURES))
        return f

    if pos_columns is None:
        if 'z' in f:
            pos_columns = ['x', 'y', 'z']
        else:
            pos_columns = ['x', 'y']

    f = f.copy()

    # Correct each column individually (subject to further optimization,
    # by assuming sub-pixel bias is equal across certain dimensions).
    for col in pos_columns:
        # Get the values as a numpy array
        x = np.array(f[col])

        # Get the sub-pixel values
        spiff = x % 1

        # Mirror the values around the center of the pixel
        spiff_mirrored = np.where(spiff < 0.5, spiff, 1 - spiff)

        # Sort the values for efficient search
        spiff_sorted = np.sort(spiff_mirrored)

        # Reverse any sub-pixel bias
        spiff_corrected_low = np.searchsorted(spiff_sorted, spiff) / len(x) / 2
        spiff_corrected_high = 1 - np.searchsorted(spiff_sorted, 1 - spiff) / len(x) / 2
        spiff_corrected = np.where(spiff < 0.5, spiff_corrected_low, spiff_corrected_high)

        # Add the sub-pixel value back to the original pixel position
        x_corrected = np.floor(x) + spiff_corrected
        f[col] = x_corrected

    return f
