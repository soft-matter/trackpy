from scipy.spatial import cKDTree
import random
import numpy as np
from warnings import warn



def pairCorrelationKDTree2D(feat, cutoff, fraction = 1., dr = .5, p_indexes = None, ndensity=None, boundary = None): 
    """   
    Calculate the pair correlation function in 2 dimensions.

    Parameters
    ----------
    feat : Pandas DataFrame
        DataFrame containing the x and y coordinates of particles
    cutoff : float
        Maximum distance to calculate g(r)
    fration : float, optional
        The fraction of particles to calculate g(r) with. May be used to increase speed of function. Particles selected at random.
    dr : float, optional
        The bin width
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
        xmin, xmax, ymin, ymax =  feat.x.min(), feat.x.max(), feat.y.min(), feat.y.max()
    else:
       xmin, xmax, ymin, ymax = boundary
       feat = feat[(feat.x > xmin) & (feat.x < xmax) & (feat.y > ymin) & (feat.y < ymax)] # Disregard all particles outside the bounding box

    if ndensity is None:
        ndensity = feat.x.count() / ((feat.x.max() - feat.x.min()) * (feat.y.max() - feat.y.min())) #  particle packing density 
        warn("Rectangular homogenous packing is assumed to calculate particle density.")

    if p_indexes is None:
        p_indexes = random.sample(range(len(feat)), int(fraction*len(feat)))  # grab random sample of particles

    r_edges = np.arange(dr, cutoff + dr, dr)  # radii bins to search for particles
    g_r = np.zeros(len(r_edges) - 1) 
    max_p_count =  int(np.pi * (r_edges.max() + dr)**2 * ndensity * 10)  # upper bound for neighborhood particle count
    ckdtree = cKDTree(feat[['x', 'y']])  # initialize kdtree for fast neighbor search
    points = feat.as_matrix(['x', 'y'])  # Convert pandas dataframe to numpy array for fast indexing
        
    # For edge handling, two techniques are used. If a particle is near only one edge, the area of the search ring r+dr is 
    # caluclated analytically via 1 - arccos(d / r ) / pi, where d is the distance to the wall
    # If the particle is near two or more walls, a ring of points is generated around the particle, and a mask
    # is applied to find the the number of points within the boundary, giving an estimate of the area
    # Below, rings of size r + dr  for all r in r_edges are generated and cahched for later use to speed up computation
    n = 100 #FIXME. n * layers = the number of points in the ring
    layers = 5
    refx, refy = _points_ring(r_edges, dr, layers, n)

    for idx in p_indexes:
        dist, idxs = ckdtree.query(points[idx], k=max_p_count, distance_upper_bound=cutoff)
        area = np.ones(len(r_edges)) * dr * r_edges * 2 * np.pi

        # Find the number of edge collisions at each radii
        collisions = _num_wall_collisions(points[idx], r_edges, xmin, xmax, ymin, ymax)

        # If some disk will collide with the wall, we need to implement edge handling
        if collisions.max() > 0:

            # Use analyitcal solution to find area of disks cut off by one wall
            d = _distances_to_wall(points[idx], xmin, xmax, ymin, ymax).min() #grab the distance to the closest wall

            inx = np.where(collisions == 1)[0]
            area[inx] *= 1 - np.arccos(d / (r_edges[inx])) / np.pi 
        
            # If disk is cutoff by 2 or more walls, generate a bunch of points and use a mask to estimate the area within the boundaries
            inx = np.where(collisions >= 2)[0]
            x = refx[inx] + points[idx,0]
            y = refy[inx] + points[idx,1]
            mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
            area[inx] *= mask.sum(axis=1, dtype='float') / len(refx[0])
            

        g_r +=  np.histogram(dist, bins = r_edges)[0] / (area[:-1])  
    g_r /= (ndensity * len(p_indexes))
    return r_edges, g_r

def _num_wall_collisions(point, radius, xmin, xmax, ymin, ymax):
    collisions = (point[0] + radius >= xmax).astype(int) + (point[0] - radius <= xmin).astype(int) + \
                 (point[1] + radius >= ymax).astype(int) + (point[1] - radius <= ymin).astype(int)

    return collisions
    
def _distances_to_wall(point, xmin, xmax, ymin, ymax): 
    return np.array([point[0]-xmin, xmax-point[0], point[1]-ymin, ymax-point[1]])

def _points_ring(r_edges, dr, layers, n):
    """Returns x, y array of points comprising shells extending from r to r_dr. layers determines how many concentric layers are in each shell,
        and n determines the number of points in each layer"""

    refx=np.empty((len(r_edges), n*layers))
    refy=refx.copy()
    for index, r in enumerate(r_edges): 
        theta = np.linspace(0, 2*np.pi, n)
        theta = theta.repeat(layers).reshape((len(theta), layers))
        x = np.cos(theta) * np.linspace(r, r+dr, layers)
        y = np.sin(theta) * np.linspace(r, r+dr, layers)
        refx[index] = x.reshape(n*layers)
        refy[index] = y.reshape(n*layers)

    return refx, refy



def pairCorrelationKDTree2Dold(feat, fraction = 1, dr = .5, cutoff = 20, p_indexes=None, ndensity=None, boundary = None):
    if boundary is None:
        warn("Rectangular packing is assumed. Boundaries are determined by edge particles.")
        xmin, xmax, ymin, ymax =  feat.x.min(), feat.x.max(), feat.y.min(), feat.y.max()
    else:
       xmin, xmax, ymin, ymax = boundary
       feat = feat[(feat.x > xmin) & (feat.x < xmax) & (feat.y > ymin) & (feat.y < ymax)] # Disregard all particles outside the bounding box

    if ndensity is None:
        ndensity = feat.x.count() / ((feat.x.max() - feat.x.min()) * (feat.y.max() - feat.y.min())) #  particle packing density 
        warn("Rectangular homogenous packing is assumed to calculate particle density.")

    if p_indexes is None:
        p_indexes = random.sample(range(len(feat)), int(fraction*len(feat)))  # grab random sample of particles

    r_edges = np.arange(0, cutoff + dr, dr)  # radii bins to search for particles
    print r_edges
    g_r = np.zeros(len(r_edges) - 1) 
    max_p_count =  int(np.pi * (r_edges.max() + dr)**2 * ndensity * 10)  # upper bound for neighborhood particle count
    ckdtree = cKDTree(feat[['x', 'y']])  # initialize kdtree for fast neighbor search
    points = feat.as_matrix(['x', 'y'])  # Convert pandas dataframe to numpy array for fast indexing
        
    # For edge handling, two techniques are used. If a particle is near only one edge, the area of the search ring r+dr is 
    # caluclated analytically via 1 - arccos(d / r ) / pi, where d is the distance to the wall
    # If the particle is near two or more walls, a ring of points is generated around the particle, and a mask
    # is applied to find the the number of points within the boundary, giving an estimate of the area
    # Below, rings of size r + dr  for all r in r_edges are generated and cahched for later use to speed up computation
    n = 100 #FIXME. n * layers = the number of points in the ring
    layers = 5
    refx, refy = _points_ring(r_edges, dr, layers, n)

    for idx in p_indexes:
        dist, idxs = ckdtree.query(points[idx], k=max_p_count, distance_upper_bound=cutoff)
        area = np.ones(len(r_edges)) * dr * r_edges * 2 * np.pi               
        g_r +=  np.histogram(dist, bins = r_edges)[0] / (area[:-1])  
    g_r /= (ndensity * len(p_indexes))
    return r_edges, g_r



def pairCorrelationKDTree3D(feat, fraction = .10, dr = .5, cutoff = 20, p_indexes=None, ndensity=None, boundary = None):
    if boundary is None:
        xmin = feat.x.min()
        xmax = feat.x.max()
        ymin = feat.y.min()
        ymax = feat.y.max()
        zmin = feat.z.min()
        zmax = feat.z.max()
    else:
        xmin = boundary[0]
        xmax = boundary[1]
        ymin = boundary[2]
        ymax = boundary[3]
        zmin = boundary[4]
        zmax = boundary[5]

    if ndensity is None:
        ndensity = feat.x.count() / ((feat.x.max() - feat.x.min()) * (feat.y.max() - feat.y.min()) * (feat.z.max() - feat.z.min()) ) #particle packing density 
        warn("Rectangular homogenous packing is assumed. Must specify p density if otherwise")

    if p_indexes is None:
        p_indexes = random.sample(range(len(feat)), int(fraction*len(feat))) #grab random sample of particles
    
    
    r_edges = np.arange(dr, cutoff + 2*dr, dr) #radii to search for particles
    g_r = np.zeros(len(r_edges) - 1) 

    max_p_count =  int((4.0 / 3.0) * np.pi * (r_edges.max() + dr)**3 * ndensity * 10) #upper bound for neighborhood particle count
    kdtree = cKDTree(feat[['x', 'y']])#initialize kdtree
    points = feat.as_matrix(['x', 'y'])
        
    #create reference unit disk
    n = 100
    x = np.linspace(-1, 1, n)
    y = x.copy()
    x = np.tile(x, n)
    y = y.repeat(n)
    mask1 = x**2 + y**2 + z**2 <= 1
    mask2 = x**2 + y**2 + z**2 >= 1 - dr
    refx = x[mask1 & mask2]
    refy = y[mask1 & mask2]
    refz = z[mask1 & mask2]

    for idx in p_indexes:
        dist, idxs = kdtree.query(points[idx], k=max_p_count, distance_upper_bound=(cutoff+dr))


        area = np.ones(len(r_edges))
        
        #check distance of particle to closest edge
        for index, r in enumerate(r_edges):        
            #shift reference circle to the correct point in the frame
            x = (r * refx + points[idx,0]) 
            y = (r * refy + points[idx,1]) 
            z = (r * refz + points[idx,2]) 
            mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax) & (z >= zmin) & (y <= zmax)
            area[index] = (float(np.count_nonzero(mask) / len(refx))) * r
        g_r +=  np.histogram(dist, bins = r_edges)[0] / area[:-1]
        
    g_r[(g_r == np.inf)] = np.NaN    
    g_r /= np.linalg.norm(g_r)
    
    return r_edges, g_r






   









