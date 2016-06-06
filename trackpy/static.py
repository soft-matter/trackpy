from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
from scipy.spatial import cKDTree
import numpy as np
from pandas import DataFrame
from warnings import warn

# Maximum number of elements in the array of all distances.
# Should be roughly (bytes of available memory)/16
MAX_ARRAY_SIZE = 1e8


def proximity(features, pos_columns=None):
    """Find the distance to each feature's nearest neighbor.

    Parameters
    ----------
    features : DataFrame
    pos_columns : list of column names
        ['x', 'y'] by default

    Returns
    -------
    proximity : DataFrame
        distance to each particle's nearest neighbor,
        indexed by particle if 'particle' column is present in input

    Examples
    --------
    Find the proximity of each particle to its nearest neighbor in every frame.

    >>> prox = t.groupby('frame').apply(proximity).reset_index()
    >>> avg_prox = prox.groupby('particle')['proximity'].mean()

    And filter the trajectories...

    >>> particle_nos = avg_prox[avg_prox > 20].index
    >>> t_filtered = t[t['particle'].isin(particle_nos)]
    """
    if pos_columns is None:
        pos_columns = ['x', 'y']
    leaf_size = max(1, int(np.round(np.log10(len(features)))))
    tree = cKDTree(features[pos_columns].copy(), leaf_size)
    proximity = tree.query(tree.data, 2)[0][:, 1]
    result = DataFrame({'proximity': proximity})
    if 'particle' in features:
        result.set_index(features['particle'], inplace=True)
    return result


def pair_correlation_2d(feat, cutoff, fraction=1., dr=.5, p_indices=None,
                        ndensity=None, boundary=None, handle_edge=True,
                        max_rel_ndensity=10):
    """Calculate the pair correlation function in 2 dimensions.

    Parameters
    ----------
    feat : Pandas DataFrame
        DataFrame containing the x and y coordinates of particles
    cutoff : float
        Maximum distance to calculate g(r)
    fraction : float, optional
        The fraction of particles to calculate g(r) with. May be used to
        increase speed of function. Particles selected at random.
    dr : float, optional
        The bin width
    p_indices : list or ndarray, optional
        Only consider a pair of particles if one of them is in 'p_indices'.
        Uses zero-based indexing, regardless of how 'feat' is indexed.
    ndensity : float, optional
        Density of particle packing. If not specified, density will be
        calculated assuming rectangular homogeneous arrangement.
    boundary : tuple, optional
        Tuple specifying rectangular prism boundary of particles (xmin, xmax,
        ymin, ymax). Must be floats. Default is to assume a rectangular packing.
        Boundaries are determined by edge particles.
    handle_edge : boolean, optional
        If true, compensate for reduced area around particles near the edges.
    max_rel_ndensity : number, optional
        The relative maximum density deviation, used to estimate the maximum
        number of neighbours. Lower numbers increase performance, until the
        method fails because there are more neighbours than expected.

    Returns
    -------
    r_edges : array
        The bin edges, with 1 more element than g_r.
    g_r : array
        The values of g_r.
    """

    if boundary is None:
        xmin, xmax, ymin, ymax = (feat.x.min(), feat.x.max(),
                                  feat.y.min(), feat.y.max())
    else:
        xmin, xmax, ymin, ymax = boundary

        # Disregard all particles outside the bounding box
        feat = feat[(feat.x >= xmin) & (feat.x <= xmax) &
                    (feat.y >= ymin) & (feat.y <= ymax)]

    if ndensity is None:  # particle packing density
        ndensity = (feat.x.count() - 1) / ((xmax - xmin) * (ymax - ymin))

    if p_indices is None:
        if fraction == 1.:
            p_indices = slice(len(feat))
        else:  # grab random sample of particles
            p_indices = np.random.randint(0, len(feat),
                                          int(fraction * len(feat)))

    # radii bins to search for particles
    r_edges = np.arange(0, cutoff + dr, dr)

    # initialize kdtree for fast neighbor search
    ckdtree = cKDTree(feat[['x', 'y']])
    pos = ckdtree.data[p_indices]

    # Estimate upper bound for neighborhood particle count
    max_p_count = int(np.pi * (r_edges.max() + dr)**2 *
                      ndensity * max_rel_ndensity)
    # Protect against too large memory usage
    if len(pos) * max_p_count > MAX_ARRAY_SIZE:
          raise MemoryError('The distance array will be larger than the maximum '
                          'allowed size. Please reduce the cutoff or '
                          'max_rel_ndensity. Or run the analysis on a fraction '
                          'of the features using the fraction parameter.')

    dist, idxs = ckdtree.query(pos, k=max_p_count, distance_upper_bound=cutoff)
    if np.any(np.isfinite(dist[:, -1])):
        raise RuntimeError("There are too many particle pairs per particle. "
                           "Apparently, density fluctuations are larger than "
                           "max_rel_ndensity. Please increase it.")

    # drop zero and infinite dist values
    mask = (dist > 0) & np.isfinite(dist)
    dist = dist[mask]

    if handle_edge:
        pos_repeated = pos[:, np.newaxis].repeat(max_p_count, axis=1)[mask]
        arclen = arclen_2d_bounded(dist, pos_repeated,
                                   np.array([[xmin, xmax], [ymin, ymax]]))
    else:
        arclen = 2*np.pi*dist
    g_r = np.histogram(dist, bins=r_edges, weights=1/arclen)[0]

    return r_edges, g_r / (ndensity * len(pos) * dr)


def pair_correlation_3d(feat, cutoff, fraction=1., dr=.5, p_indices=None,
                        ndensity=None, boundary=None, handle_edge=True,
                        max_rel_ndensity=10):
    """Calculate the pair correlation function in 3 dimensions.

    Parameters
    ----------
    feat : Pandas DataFrame
        DataFrame containing the x, y and z coordinates of particles
    cutoff : float
        Maximum distance to calculate g(r)
    fraction : float, optional
        The fraction of particles to calculate g(r) with. May be used to
        increase speed of function. Particles selected at random.
    dr : float, optional
        The bin width
    p_indices : list or ndarray, optional
        Only consider a pair of particles if one of them is in 'p_indices'.
        Uses zero-based indexing, regardless of how 'feat' is indexed.
    ndensity : float, optional
        Density of particle packing. If not specified, density will be
        calculated assuming rectangular homogeneous arrangement.
    boundary : tuple, optional
        Tuple specifying rectangular boundary of particles (xmin, xmax,
        ymin, ymax, zmin, zmax). Must be floats. Default is to assume a
        rectangular packing. Boundaries are determined by edge particles.
    handle_edge : boolean, optional
        If true, compensate for reduced volume around particles near the edges.
    max_rel_ndensity : number, optional
        The relative maximum density deviation, used to estimate the maximum
        number of neighbours. Lower numbers increase performance, until the
        method fails because there are more neighbours than expected.

    Returns
    -------
    r_edges : array
        The bin edges, with 1 more element than g_r.
    g_r : array
        The values of g_r.
    """

    if boundary is None:
        xmin, xmax, ymin, ymax, zmin, zmax = (feat.x.min(), feat.x.max(),
                                              feat.y.min(), feat.y.max(),
                                              feat.z.min(), feat.z.max())
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = boundary

        # Disregard all particles outside the bounding box
        feat = feat[(feat.x >= xmin) & (feat.x <= xmax) &
                    (feat.y >= ymin) & (feat.y <= ymax) &
                    (feat.z >= zmin) & (feat.z <= zmax)]

    if ndensity is None:  # particle packing density
        ndensity = (feat.x.count() - 1) / \
                   ((xmax - xmin) * (ymax - ymin) * (zmax - zmin))

    if p_indices is None:
        if fraction == 1.:
            p_indices = slice(len(feat))
        else:  # grab random sample of particles
            p_indices = np.random.randint(0, len(feat),
                                          int(fraction * len(feat)))

    # radii bins to search for particles
    r_edges = np.arange(0, cutoff + dr, dr)

    # initialize kdtree for fast neighbor search
    ckdtree = cKDTree(feat[['x', 'y', 'z']])
    pos = ckdtree.data[p_indices]

    # Estimate upper bound for neighborhood particle count
    max_p_count = int((4./3.) * np.pi * (r_edges.max() + dr)**3 *
                      ndensity * max_rel_ndensity)
    # Protect against too large memory usage
    if len(pos) * max_p_count > MAX_ARRAY_SIZE:
        raise MemoryError('The distance array will be larger than the maximum '
                          'allowed size. Please reduce the cutoff or '
                          'max_rel_ndensity. Or run the analysis on a fraction '
                          'of the features using the fraction parameter.')

    dist, idxs = ckdtree.query(pos, k=max_p_count, distance_upper_bound=cutoff)
    if np.any(np.isfinite(dist[:, -1])):
        raise RuntimeError("There are too many particle pairs in the frame. "
                           "Please reduce the cutoff distance, increase "
                           "max_rel_ndensity, or use a fraction.")

    # drop zero and infinite dist values
    mask = (dist > 0) & np.isfinite(dist)
    dist = dist[mask]

    if handle_edge:
        pos_repeated = pos[:, np.newaxis].repeat(max_p_count, axis=1)[mask]
        area = area_3d_bounded(dist, pos_repeated,
                               np.array([[xmin, xmax], [ymin, ymax],
                                         [zmin, zmax]]))
    else:
        area = 4*np.pi*dist**2
    g_r = np.histogram(dist, bins=r_edges, weights=1/area)[0]

    return r_edges, g_r / (ndensity * len(pos) * dr)


def circle_cap_arclen(h, r):
    """ Length of a circle arc of circle with radius R that is bounded by
    a straight line `h` from the origin. h >= 0, h < R"""
    return 2*r*np.arccos(h / r)


def circle_corner_arclen(h1, h2, r):
    """ Length of a circle arc of circle with radius R that is bounded by
    two perpendicular straight lines `h1` and `h2` from the origin.
    h1**2 + h2**2 < R**2
    h1 >= R
    h2 >= R
    """
    return r*(np.arccos(h2 / r) - np.arcsin(h1 / r))


def sphere_cap_area(h, r):
    """ Area of a sphere cap of sphere with radius R that is bounded by
    a flat plane `h` from the origin. h >= 0, h < R"""
    return 2*np.pi*r*(r-h)


def sphere_edge_area(x, y, r):
    """ Area of a sphere 'edge' of sphere with radius R that is bounded by
    two perpendicular flat planes `h0`, `h1` from the origin. h >= 0, h < R"""
    p = np.sqrt(r**2 - x**2 - y**2)
    A = (r - x - y)*np.pi - 2*r*np.arctan(x*y/(p*r)) + \
        2*x*np.arctan(y/p) + 2*y*np.arctan(x/p)
    return A*r


def sphere_corner_area(x, y, z, r):
    """ Area of a sphere 'corner' of sphere with radius R that is bounded by
    three perpendicular flat planes `h0`, `h1`, `h2` from the origin. """
    pxy = np.sqrt(r**2 - x**2 - y**2)
    pyz = np.sqrt(r**2 - y**2 - z**2)
    pxz = np.sqrt(r**2 - x**2 - z**2)
    A = np.pi*(r - x - y - z)/2 + \
        x*(np.arctan(y/pxy) + np.arctan(z/pxz)) - r*np.arctan(y*z/(r*pyz)) + \
        y*(np.arctan(x/pxy) + np.arctan(z/pyz)) - r*np.arctan(x*z/(r*pxz)) + \
        z*(np.arctan(x/pxz) + np.arctan(y/pyz)) - r*np.arctan(x*y/(r*pxy))
    return A*r


def _protect_mask(mask):
    """ Boolean masks with length 1 may give a problem in the following syntax:
    array[mask] = 2 * array[mask]. Replace [True] by 0 and return None when
    the operation should be skipped."""
    if mask.size == 0:
        return None  # []
    elif mask.size > 1:
        return mask  # mask with len > 1
    if mask.ravel()[0]:
        return 0     # [True]
    else:
        return None  # [False]


def arclen_2d_bounded(dist, pos, box):
    arclen = 2*np.pi*dist

    h = np.array([pos[:, 0] - box[0, 0], box[0, 1] - pos[:, 0],
                  pos[:, 1] - box[1, 0], box[1, 1] - pos[:, 1]])

    for h0 in h:
        mask = _protect_mask(h0 < dist)
        if mask is None:
            continue
        arclen[mask] -= circle_cap_arclen(h0[mask], dist[mask])

    for h1, h2 in [[0, 2], [0, 3], [1, 2], [1, 3]]:  # adjacent sides
        mask = _protect_mask(h[h1]**2 + h[h2]**2 < dist**2)
        if mask is None:
            continue
        arclen[mask] += circle_corner_arclen(h[h1, mask], h[h2, mask],
                                             dist[mask])

    arclen[arclen < 10**-5 * dist] = np.nan
    return arclen


def area_3d_bounded(dist, pos, box):
    """ Calculated using the surface area of a sphere equidistant
    to a certain point.

    When the sphere is truncated by the box boundaries, this distance
    is subtracted using the formula for the sphere cap surface. We
    calculate this by defining h = the distance from point to box edge.

    When for instance sphere is bounded by the top and right boundaries,
    the area in the edge may be counted double. This is the case when
    h1**2 + h2**2 < R**2. This double counted area is calculated
    and added if necessary.

    When the sphere is bounded by three adjacant boundaries,
    the area in the corner may be subtracted double. This is the case when
    h1**2 + h2**2 + h3**2 < R**2. This double counted area is calculated
    and added if necessary.

    The result is the sum of the weights of pos0 and pos1."""

    area = 4*np.pi*dist**2

    h = np.array([pos[:, 0] - box[0, 0], box[0, 1] - pos[:, 0],
                  pos[:, 1] - box[1, 0], box[1, 1] - pos[:, 1],
                  pos[:, 2] - box[2, 0], box[2, 1] - pos[:, 2]])

    for h0 in h:
        mask = _protect_mask(h0 < dist)
        if mask is None:
            continue
        area[mask] -= sphere_cap_area(h0[mask], dist[mask])

    for h1, h2 in [[0, 2], [0, 3], [0, 4], [0, 5],
                   [1, 2], [1, 3], [1, 4], [1, 5],
                   [2, 4], [2, 5], [3, 4], [3, 5]]:  # 2 adjacent sides
        mask = _protect_mask(h[h1]**2 + h[h2]**2 < dist**2)
        if mask is None:
            continue
        area[mask] += sphere_edge_area(h[h1, mask], h[h2, mask],
                                       dist[mask])

    for h1, h2, h3 in [[0, 2, 4], [0, 2, 5], [0, 3, 4], [0, 3, 5], [1, 2, 4],
                       [1, 2, 5], [1, 3, 4], [1, 3, 5]]:  # 3 adjacent sides
        mask = _protect_mask(h[h1]**2 + h[h2]**2 + h[h3]**2 < dist**2)
        if mask is None:
            continue
        area[mask] -= sphere_corner_area(h[h1, mask], h[h2, mask],
                                         h[h3, mask], dist[mask])

    area[area < 10**-7 * dist**2] = np.nan

    return area
