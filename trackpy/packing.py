from scipy.spatial import cKDTree
import random
import numpy as np
from warnings import warn



def pairCorrelation2D(feat, fraction = 1., dr = .5, cutoff = 20., ndensity=None, boundary = None):
    """
    Calculate the pair correlation function in 2 dimensions.

    Parameters
    ----------
    feat : Pandas DataFrame
        DataFrame containing the x and y coordinates of particles
    fration : float, optional
        The fraction of particles to calculate g(r) with. May be used to increase speed of function. Particles selected at random.
    dr : float, optional
        The bin width
    cutoff : float, optional
        Maximum distance to calculate g(r)
    ndensity : float, optional
        Density of particle packing. If not specified, density will be calculated assuming rectangular homogenous arangement
    boundary : Tuple, optional
        Tuple specifying rectangular boundary of partcicles (xmin, xmax, ymin, ymax)

    Returns
    -------
    r_edges : array
        Return the bin edges
    g_r : array
        The values of g_r
    """

    if boundary is None:
        warn("Rectangular packing is assumed. Boundaries are determined by edge particles.")
        xmin = feat.x.min()
        xmax = feat.x.max()
        ymin = feat.y.min()
        ymax = feat.y.max()
    else:
        xmin = boundary[0]
        xmax = boundary[1]
        ymin = boundary[2]
        ymax = boundary[3]

    if ndensity is None:
        ndensity = feat.x.count() / ((feat.x.max() - feat.x.min()) * (feat.y.max() - feat.y.min())) #particle packing density
        warn("Rectangular homogenous packing is assumed to calculate particle density.")


    p_indexes = random.sample(range(len(feat)), int(fraction*len(feat))) #grab random sample of particles
    r_edges = np.arange(dr, cutoff + 2*dr, dr) #radii to search for particles
    g_r = np.zeros(len(r_edges) - 1)
    max_p_count =  int(np.pi * (r_edges.max() + dr)**2 * ndensity * 10) #upper bound for neighborhood particle count
    ckdtree = cKDTree(feat[['x', 'y']])#initialize kdtree
    points = feat.as_matrix(['x', 'y'])

    #create reference disks of all radii in r_edges
    n = 50
    x = np.linspace(-1, 1, n)
    y = x.copy()
    x = np.tile(x, n)
    y = y.repeat(n)
    mask1 = x**2 + y**2 <= 1
    mask2 = x**2 + y**2 >= 1 - dr
    x = x[mask1 & mask2]
    y = y[mask1 & mask2]
    refx = np.ones((len(r_edges), len(x)))
    refx[:] = x
    refx *= r_edges.repeat(len(x)).reshape((len(r_edges), len(x)))
    refy = np.ones((len(r_edges), len(y)))
    refy[:] = y
    refy *= r_edges.repeat(len(y)).reshape((len(r_edges), len(y)))

    for idx in p_indexes:
        dist, idxs = ckdtree.query(points[idx], k=max_p_count, distance_upper_bound=(cutoff+dr))
        area = np.ones(len(r_edges))

        ##Find the number of edge collisions at each radii
        collisions = num_wall_collisions(points[idx], r_edges, xmin, xmax, ymin, ymax)

        #if some disk will collide with the wall, we need to implement edge handling
        if collisions.max() > 0:

            #Use analyitcal solution to find area of disks cut off by one wall
            d = distances_to_wall(points[idx], xmin, xmax, ymin, ymax).min() #grab the distance to the closest wall
            inx = np.where(collisions == 1)[0]
            area[inx] = (np.pi - np.arccos(d / r_edges[inx])) / np.pi

            #If disk is cutoff by 2 or more walls, generate a bunch of points and use a mask to estimate the area within the boundaries
            inx = np.where(collisions >= 2)[0]
            x = refx[inx] + points[idx,0]
            y = refy[inx] + points[idx,1]
            mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
            area[inx] = mask.sum(axis=1, dtype='float') / len(refx[0])

        g_r +=  np.histogram(dist, bins = r_edges)[0] / (area[:-1]  * r_edges[:-1])

    g_r /= np.linalg.norm(g_r)

    return r_edges, g_r

def num_wall_collisions(point, radius, xmin, xmax, ymin, ymax):
    collisions = (point[0] + radius >= xmax).astype(int) +  (point[0] - radius <= xmin) +  \
                 (point[1] + radius >= ymax) +  (point[1] - radius <= ymin)
    return collisions

def distances_to_wall(point, xmin, xmax, ymin, ymax):
    dist = np.zeros(4)
    dist[0] = point[0] - xmin
    dist[1] = xmax - point[0]
    dist[2] = point[1] - ymin
    dist[3] = ymax - point[1]
    return dist















