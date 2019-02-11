import numpy as np
import pandas as pd

from ..utils import (validate_tuple, guess_pos_columns, default_pos_columns)


def refine_brightfield_ring(image, radius, coords_df, pos_columns=None):
    """Find the center of mass of a brightfield feature starting from an
    estimate.

    Parameters
    ----------
    image : array (any dimension)
        processed image, used for locating center of mass
    coords_df : DataFrame
        estimated positions
    pos_columns: list of strings, optional
        Column names that contain the position coordinates.
        Defaults to ``['y', 'x']`` or ``['z', 'y', 'x']``, if ``'z'`` exists.
    """
    if pos_columns is None:
        pos_columns = guess_pos_columns(coords_df)

    radius = validate_tuple(radius, image.ndim)

    if pos_columns is None:
        pos_columns = default_pos_columns(image.ndim)

    columns = pos_columns + ['size']

    if len(coords_df) == 0:
        return pd.DataFrame(columns=columns)

    refined = coords_df

    return refined
