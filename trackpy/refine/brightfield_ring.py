"""
Detect particles in brightfield mode by tracking a ring of dark pixels around a
bright interior part. Based on https://github.com/caspervdw/circletracking
"""
import numpy as np
import pandas as pd
import warnings

from scipy.ndimage import map_coordinates
from scipy import stats

from ..utils import (validate_tuple, guess_pos_columns, default_pos_columns,
                     stats_mode_scalar)


def refine_brightfield_ring(image, radius, coords_df, pos_columns=None,
                            rad_range=None, **kwargs):
    """Find the center of mass of a brightfield feature starting from an
    estimate.

    Parameters
    ----------
    image : array (any dimension)
        processed image, used for locating center of mass
    radius : tuple(int, int)
        the estimated radii of the feature. Note: only the first value is used
        for now.
    coords_df : Series([x, y])
        estimated position of the feature
    pos_columns: list of strings, optional
        Column names that contain the position coordinates.
        Defaults to ``['y', 'x']`` or ``['z', 'y', 'x']``, if ``'z'`` exists.
    rad_range : tuple(float, float)
        The initial search range and final search range in the last iteration
    **kwargs:
        Passed to the min_edge function.

    Returns
    -------
    Series([x, y, r])
        where r means the radius of the fitted circle of dark pixels around
        the bright interior of the particle. Returns None on failure.
    """
    if not isinstance(coords_df, pd.core.series.Series) or len(coords_df) != 2:
        raise ValueError("Refine brightfield ring only supports a Series" +
                         " of 1 particle with values x, y")

    if pos_columns is None:
        pos_columns = guess_pos_columns(coords_df)

    radius = validate_tuple(radius, image.ndim)

    if pos_columns is None:
        pos_columns = default_pos_columns(image.ndim)

    r = radius[0]
    result = _refine_brightfield_ring(image, r, coords_df, rad_range=rad_range,
                                      **kwargs)

    refined_r, refined_x, refined_y = result

    if refined_r is None or refined_y is None or refined_x is None:
        return None

    coords_df['x'] = refined_x
    coords_df['y'] = refined_y
    coords_df['r'] = refined_r

    return coords_df

def _refine_brightfield_ring(image, radius, coords_df, min_points_frac=0.35,
                             max_ev=10, rad_range=None, max_r_dev=0.5, **kwargs):
    """Find the center of mass of a brightfield feature starting from an
    estimate.

    Parameters
    ----------
    image : array (any dimension)
        processed image, used for locating center of mass
    radius : int
        the estimated radius of the feature
    coords_df : DataFrame
        estimated positions
    threshold : float
        the relative threshold to use to find the edge
    max_dev : float
        points further away than `max_dev` pixels from the fit of the circle
        are discarded.
    min_points_frac : float
        the minimum fraction of circumference found for a fit to be considered
        accurate enough
    max_ev : int
        the maximum number of refinement steps
    rad_range : tuple(float, float)
        The search range
    max_r_dev : float
        The maximum relative difference in the true and tracked radius.
        The condition abs(Rtrue - Rtracked) / Rtrue < max_r_dev should hold,
        otherwise the refinement is retried.
    min_percentile : float
        The percentile (0.0-100.0) below which pixels are considered as part of
        the dark ring around the feature. Use lower values for features that
        have sharper, more defined edges.

    Returns
    -------
    r : float
        the fitted radius of the feature
    x : float
        the fitted x coordinate of the feature
    y : float
        the fitted y coordinate of the feature
    """
    coords_df = coords_df.astype(float)
    if max_ev == 0:
        # final iteration reached
        return radius, coords_df['x'], coords_df['y']

    radius = float(radius)
    if rad_range is None:
        rad_range = (-radius, radius)

    # Get intensity in spline representation
    coords = (radius, radius, float(coords_df['y']), float(coords_df['x']))
    intensity, pos, normal = _unwrap_ellipse(image, coords, rad_range)

    # Find the coordinates of the edge
    r_dev = _min_edge(intensity, **kwargs)
    r_dev += rad_range[0]
    if np.sum(~np.isnan(r_dev))/len(r_dev) < min_points_frac:
        return _retry(image, radius, coords_df, min_points_frac, max_ev,
                      rad_range, **kwargs)

    # Convert to cartesian
    coord_new = _to_cartesian(r_dev, pos, normal)

    # Fit the circle
    try:
        r, (xc, yc) = _fit_circle(coord_new)
    except np.linalg.LinAlgError:
        return _retry(image, radius, coords_df, min_points_frac, max_ev,
                      rad_range, **kwargs)
    if np.any(np.isnan([r, yc, xc])):
        return _retry(image, radius, coords_df, min_points_frac, max_ev,
                      rad_range, **kwargs)
    if not rad_range[0] < r - radius < rad_range[1]:
        return _retry(image, radius, coords_df, min_points_frac, max_ev,
                      rad_range, **kwargs)

    if np.abs(radius-r)/radius > max_r_dev:
        return _retry(image, radius, coords_df, min_points_frac, max_ev,
                      rad_range, **kwargs)

    return r, xc, yc

def _retry(image, radius, coords_df, min_points_frac, max_ev, rad_range,
           **kwargs):
    # try again with different search range
    rad_range = np.multiply(1.03, rad_range)
    max_ev -= 1
    if max_ev > 0:
        return _refine_brightfield_ring(image, radius, coords_df,
                                        min_points_frac, max_ev,
                                        rad_range, **kwargs)
    return None, None, None

def _min_edge(arr, threshold=0.45, max_dev=1, axis=1, bright_left=True,
              bright_left_factor=1.2, min_percentile=5.0):
    """ Find min value of each row """
    if axis == 0:
        arr = arr.T
    if np.issubdtype(arr.dtype, np.unsignedinteger):
        arr = arr.astype(int)

    # column numbers
    indices = np.indices(arr.shape)[1]

    # values below min_percentile% for each row
    values = np.nanpercentile(arr, min_percentile, axis=1)
    bc_values = np.repeat(values[:, np.newaxis], indices.shape[1], axis=1)

    # allow np.nan's in comparison, these are filtered later
    with np.errstate(invalid='ignore'):
        # get column numbers of lowest edge values < min_percentile%
        r_dev = np.where((arr < bc_values) & ~np.isnan(arr) & ~np.isnan(bc_values), indices, np.nan)

    # allow all np.nan slices, these are filtered later
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
        # then take the median of the column indices, ignoring nan's
        r_dev = np.nanmedian(r_dev, axis=1)

    # threshold on edge
    abs_thr = threshold * np.nanmax(arr)
    r_dev[values > abs_thr] = np.nan

    mask = ~np.isnan(r_dev)
    if np.sum(mask) == 0:
        return r_dev

    # filter by deviations from most occurring value
    max_dev = np.round(max_dev)
    most_likely = stats_mode_scalar(r_dev[mask])
    mask[mask] &= np.abs(r_dev[mask]-most_likely) > max_dev
    r_dev[mask] = np.nan

    mask = ~np.isnan(r_dev)
    if np.sum(mask) == 0:
        return r_dev

    # Check if left is brighter than right
    if bright_left:
        split_i = int(np.round(most_likely))
        left = np.nanmean(arr[:, :split_i])
        right = np.nanmean(arr[:, split_i:])
        if left < bright_left_factor * right:
            return np.array(len(r_dev)*[np.nan], dtype=float)

    return r_dev

def _fit_circle(coords):
    """ Fits a circle to datapoints using an algebraic method.

    Parameters
    ----------
    coords : numpy array of floats
        array of shape (N, 2) containing datapoints

    Returns
    -------
    radius, center

    References
    ----------
    .. [1] Bertoni B (2010) Multi-dimensional ellipsoidal fitting.
    """
    if coords.shape[0] != 2:
        raise ValueError('Input data must have two columns!')

    x = coords[0, :, np.newaxis]
    y = coords[1, :, np.newaxis]

    D = np.hstack((2 * x, 2 * y, np.ones_like(x)))

    d2 = x ** 2 + y ** 2  # the RHS of the llsq problem (y's)
    u = np.linalg.solve(np.dot(D.T, D), (np.dot(D.T, d2)))[:, 0]
    v = np.empty((6), dtype=u.dtype)

    v[:2] = -1
    v[2] = 0
    v[3:] = u

    A = np.array([[v[0], v[2], v[3]],
                  [v[2], v[1], v[4]],
                  [v[3], v[4], v[5]]])
    # find the center of the ellipse
    center = -np.linalg.solve(A[:2, :2], v[3:5])

    # translate to the center
    T = np.identity(3, dtype=A.dtype)
    T[2, :2] = center
    R = np.dot(np.dot(T, A), T.T)

    # solve the eigenproblem
    evals, evecs = np.linalg.eig(R[:2, :2] / -R[2, 2])
    radius = (np.sqrt(1 / np.abs(evals)) * np.sign(evals))

    return radius[0], center

def _to_cartesian(r_dev, pos, normal):
    """ Transform radial deviations from an ellipsoidal grid to Cartesian

    Parameters
    ----------
    r_dev : ndarray, shape (N, )
        Array containing the N radial deviations from the ellipse. r < 0 means
        inside the ellipse.
    pos : ndarray, shape (2, N)
        The N (y, x) positions of the ellipse (as given by ``ellipse_grid``)
    normal : ndarray, shape (2, N)
        The N (y, x) unit normals of the ellipse (as given by ``ellipse_grid``)
    """
    coord_new = pos + r_dev * normal
    coord_new = coord_new[:, np.isfinite(coord_new).all(0)]
    coord_new[[1,0], :] = coord_new[[0,1], :]
    return coord_new

def _unwrap_ellipse(image, params, rad_range, num_points=None, spline_order=4,
                    fill_value=np.nan):
    """ Unwraps an circular or ellipse-shaped feature into elliptic coordinates.

    Transforms an image in (y, x) space to (theta, r) space, using elliptic
    coordinates. The theta coordinate is tangential to the ellipse, the r
    coordinate is normal to the ellipse. r=0 at the ellipse: inside the ellipse,
    r < 0.

    Parameters
    ----------
    image : ndarray, 2d
    params : (yr, xr, yc, xc)
    rad_range : tuple
        A tuple defining the range of r to interpolate.
    num_points : number, optional
        The number of ``theta`` values. By default, this equals the
        ellipse circumference: approx. every pixel there is an interpolation.
    spline_order : number, optional
        The order of the spline interpolation. Default 3.
    fill_value : float
        The pixel value for pixels outside the range of the original image

    Returns
    -------
    intensity : the interpolated image in (theta, r) space
    pos : the (y, x) positions of the ellipse grid
    normal : the (y, x) unit vectors normal to the ellipse grid
    """
    yr, xr, yc, xc = params
    # compute the r coordinates
    steps = np.arange(rad_range[0], rad_range[1] + 1, 1)
    # compute the (y, x) positions and unit normals of the ellipse
    pos, normal = _ellipse_grid((yr, xr), (yc, xc), n=num_points, spacing=0.25)
    # calculate all the (y, x) coordinates on which the image interpolated.
    # this is a 3D array of shape [n_theta, n_r, 2], with 2 being y and x.
    coords = normal[:, :, np.newaxis] * steps[np.newaxis, np.newaxis, :] + \
             pos[:, :, np.newaxis]
    # interpolate the image on computed coordinates
    intensity = map_coordinates(image, coords, order=spline_order,
                                output=float, mode='constant',
                                cval=fill_value)
    return intensity, pos, normal

def _ellipse_grid(radius, center, rotation=0, skew=0, n=None, spacing=1):
    """ Returns points and normal (unit) vectors on an ellipse.

    Parameters
    ----------
    radius : tuple
        (yr, xr) the two principle radii of the ellipse
    center : tuple
        (yc, xc) the center coordinate of the ellipse
    rotation : float, optional
        angle of xr with the x-axis, in radians. Rotates clockwise in image.
    skew : float, optional
        skew: y -> y + skew * x
    n : int, optional
        number of points
    spacing : float, optional
        When `n` is not given then the spacing is determined by `spacing`.

    Returns
    -------
    two arrays of shape (2, N), being the coordinates and unit normals
    """
    yr, xr = radius
    yc, xc = center
    if n is None:
        n = int(2 * np.pi * np.sqrt((yr ** 2 + xr ** 2) / 2) / spacing)

    phi = np.linspace(-np.pi, np.pi, n, endpoint=False)
    pos = np.array([yr * np.sin(phi), xr * np.cos(phi)])

    normal = np.array([np.sin(phi) / yr, np.cos(phi) / xr])
    normal /= np.sqrt((normal ** 2).sum(0))

    mask = np.isfinite(pos).all(0) & np.isfinite(normal).all(0)
    pos = pos[:, mask]
    normal = normal[:, mask]

    if rotation != 0:
        R = np.array([[np.cos(rotation), np.sin(rotation)],
                      [-np.sin(rotation), np.cos(rotation)]])
        pos = np.dot(pos.T, R).T
    elif skew != 0:
        pos[0] += pos[1] * skew

    # translate
    pos[0] += yc
    pos[1] += xc
    return pos, normal  # both in y_list, x_list format
