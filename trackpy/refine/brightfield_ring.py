import numpy as np
import pandas as pd

from scipy.ndimage import map_coordinates

from ..utils import (validate_tuple, guess_pos_columns, default_pos_columns)


def refine_brightfield_ring(image, radius, coords_df, pos_columns=None):
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

    columns = pos_columns + ['r']

    r = radius[0]
    refined_r, refined_x, refined_y = _refine_brightfield_ring(image, r,
                                                               coords_df)

    if refined_r is None or refined_y is None or refined_x is None:
        return None

    coords_df['x'] = refined_x
    coords_df['y'] = refined_y
    coords_df['r'] = refined_r

    return coords_df

def _refine_brightfield_ring(image, radius, coords_df, threshold=0.5,
                             max_r_dev=0.5, min_points_frac=0.35, max_ev=10,
                             rad_range=None):
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
    max_r_dev : float
        points further away than `max_r_dev`*`radius` pixels from the first fit
        of the circle are discarded. Decreased in consecutive iterations.
    min_points_frac : float
        the minimum fraction of circumference found for a fit to be considered
        accurate enough
    max_ev : int
        the maximum number of refinement steps
    rad_range : tuple
        the searching range (is reduced in consecutive iterations)

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

    first_rad_range = 2.0
    final_rad_range = 1.0

    if rad_range is None:
        rad_range = (-first_rad_range*radius, first_rad_range*radius)

    # Get intensity in spline representation
    coords = (radius, radius, float(coords_df['y']), float(coords_df['x']))
    intensity, pos, normal = _unwrap_ellipse(image, coords, rad_range)

    # Find the coordinates of the edge
    dark_value = (np.mean(image)-np.min(image))*0.8
    bright_value = np.mean(image)+np.max(image)*0.1
    r_dev = _min_edge(intensity, dark_value, bright_value) + rad_range[0]
    if np.sum(~np.isnan(r_dev))/len(r_dev) < min_points_frac:
        return None, None, None

    # Convert to cartesian
    coord_new = _to_cartesian(r_dev, pos, normal)

    # Fit the circle
    try:
        r, (xc, yc) = _fit_circle(coord_new)
    except np.linalg.LinAlgError:
        return None, None, None
    if np.any(np.isnan([r, yc, xc])):
        return None, None, None
    if not rad_range[0] < r - radius < rad_range[1]:
        return None, None, None

    if np.abs(radius-r)/radius > 0.5:
        return None, None, None

    # calculate deviations from circle
    x, y = coord_new
    deviations2 = (np.sqrt((xc - x) ** 2 + (yc - y) ** 2) - r) ** 2
    mask = deviations2 < (r*max_r_dev) ** 2
    if np.sum(mask)/len(mask) < min_points_frac:
        return None, None, None

    if np.any(~mask):
        try:
            r, (xc, yc) = _fit_circle(coord_new[:, mask])
        except np.linalg.LinAlgError:
            return None, None, None
        if np.any(np.isnan([r, yc, xc])):
            return None, None, None

    # next refinement iteration
    max_ev -= 1
    radius = r
    coords_df['x'] = xc
    coords_df['y'] = yc
    max_r_dev *= 0.90

    # decrease search range by 10%
    rad_range = rad_range[1]*0.90
    if rad_range < final_rad_range*radius:
        rad_range = final_rad_range*radius
    rad_range = (-rad_range, rad_range)

    result = _refine_brightfield_ring(image, radius, coords_df, threshold,
                                      max_r_dev, min_points_frac, max_ev,
                                      rad_range)

    if result[0] is None or result[1] is None or result[2] is None:
        return r, xc, yc

    return result

def _min_edge(arr, dark_value, bright_value, axis=1):
    """ Find min value of each row """
    if axis == 0:
        arr = arr.T
    if np.issubdtype(arr.dtype, np.unsignedinteger):
        arr = arr.astype(np.int)

    if np.nanmax(arr) < bright_value:
        return np.array([np.nan]*arr.shape[0])

    values = np.nanmin(arr, axis=1)
    rdev = []
    for row, min_val in zip(arr, values):
        argmin = np.where(row == min_val)[0]
        if len(argmin) == 0:
            rdev.append(np.nan)
        else:
            rdev.append(np.mean(argmin))

    r_dev = np.array(rdev)
    # threshold on inner part (should be bright on the left)
    xcoords = np.tile(np.arange(0, arr.shape[1]), (arr.shape[0], 1))
    left_mask = xcoords < np.tile(r_dev[:, np.newaxis], (1, arr.shape[1]))
    mean_intensity_inside = np.mean(arr[left_mask], axis=0)
    r_dev[mean_intensity_inside < bright_value] = np.nan
    # threshold on edge
    r_dev[values > dark_value] = np.nan
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

def _unwrap_ellipse(image, params, rad_range, num_points=None, spline_order=3,
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
    pos, normal = _ellipse_grid((yr, xr), (yc, xc), n=num_points, spacing=1)
    # calculate all the (y, x) coordinates on which the image interpolated.
    # this is a 3D array of shape [n_theta, n_r, 2], with 2 being y and x.
    coords = normal[:, :, np.newaxis] * steps[np.newaxis, np.newaxis, :] + \
             pos[:, :, np.newaxis]
    # interpolate the image on computed coordinates
    intensity = map_coordinates(image, coords, order=spline_order,
                                output=np.float, mode='constant',
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
