import numpy as np
import pandas as pd
import warnings
from trackpy.find import drop_close
from trackpy.utils import validate_tuple
from trackpy.preprocessing import bandpass
try:
    from pims import Frame as _Frame
except ImportError:
    _Frame = None


def draw_point(image, pos, value):
    image[tuple(pos)] = value


def feat_gauss(r, ndim):
    """ Gaussian at r = 0. """
    return np.exp(r**2 * ndim/-2)


def feat_ring(r, ndim, thickness):
    """ Ring feature with a gaussian profile with a certain thickness."""
    return np.exp(((r-1+thickness)/thickness)**2 * ndim/-2)


def feat_hat(r, ndim, disc_size):
    """ Solid disc of size disc_size, with Gaussian smoothed borders. """
    result = np.ones_like(r)
    mask = r > disc_size
    result[mask] = np.exp(((r[mask] - disc_size)/(1 - disc_size))**2 *
                          ndim/-2)
    return result


def feat_step(r, ndim):
    """ Solid disc. """
    return (r <= 1).astype(float)


feat_disc = feat_hat
feat_dict = dict(gauss=feat_gauss, disc=feat_disc, ring=feat_ring,
                 hat=feat_hat, step=feat_step)

def draw_feature(image, position, size, max_value=None, feat_func='gauss',
                 ecc=None, mask_diameter=None, **kwargs):
    """ Draws a radial symmetric feature and adds it to the image at given
    position. The given function will be evaluated at each pixel coordinate,
    no averaging or convolution is done.

    Parameters
    ----------
    image : ndarray
        image to draw features on
    position : iterable
        coordinates of feature position
    size : number
        the size of the feature (meaning depends on feature, for feat_gauss,
        it is the radius of gyration)
    max_value : number
        maximum feature value. should be much less than the max value of the
        image dtype, to avoid pixel wrapping at overlapping features
    feat_func : {'gauss', 'disc', 'ring'} or callable
        Default: 'gauss'
        When callable is given, it should take an ndarray of radius values
        and it should return intensity values <= 1
    ecc : positive number, optional
        eccentricity of feature, defined only in 2D. Identical to setting
        diameter to (diameter / (1 - ecc), diameter * (1 - ecc))
    mask_diameter :
        defines the box that will be drawn on. Default 4 * size.
    kwargs : keyword arguments are passed to feat_func

    See also
    --------
    draw_spots
    """
    if len(position) != image.ndim:
        raise ValueError("Number of position coordinates should match image"
                         " dimensionality.")
    if not hasattr(feat_func, '__call__'):
        feat_func = feat_dict[feat_func]
    size = validate_tuple(size, image.ndim)
    if ecc is not None:
        if len(size) != 2:
            raise ValueError("Eccentricity is only defined in 2 dimensions")
        if size[0] != size[1]:
            raise ValueError("Diameter is already anisotropic; eccentricity is"
                             " not defined.")
        size = (size[0] / (1 - ecc), size[1] * (1 - ecc))
    if mask_diameter is None:
        mask_diameter = tuple([s * 4 for s in size])
    else:
        mask_diameter = validate_tuple(mask_diameter, image.ndim)
    if max_value is None:
        max_value = np.iinfo(image.dtype).max - 3
    rect = []
    vectors = []
    for (c, s, m, lim) in zip(position, size, mask_diameter, image.shape):
        if (c >= lim) or (c < 0):
            raise ValueError("Position outside of image.")
        lower_bound = max(int(np.floor(c - m / 2)), 0)
        upper_bound = min(int(np.ceil(c + m / 2 + 1)), lim)
        rect.append(slice(lower_bound, upper_bound))
        vectors.append(np.arange(lower_bound - c, upper_bound - c) / s)
    coords = np.meshgrid(*vectors, indexing='ij')
    r = np.sqrt(np.sum(np.array(coords)**2, axis=0))
    spot = max_value * feat_func(r, ndim=image.ndim, **kwargs)
    image[tuple(rect)] += spot.astype(image.dtype)


def gen_random_locations(shape, count, margin=0):
    """ Generates `count` number of positions within `shape`. If a `margin` is
    given, positions will be inside this margin. Margin may be tuple-valued.
    """
    margin = validate_tuple(margin, len(shape))
    np.random.seed(0)
    pos = [np.random.randint(round(m), round(s - m), count)
           for (s, m) in zip(shape, margin)]
    return np.array(pos).T


def gen_connected_locations(shape, count, separation, margin=0):
    """ Generates `count` number of positions within `shape` that are touching.
    If a `margin` is given, positions will be inside this margin. Margin may be
    tuple-valued.  """
    margin = validate_tuple(margin, len(shape))
    center_pos = margin + np.round(np.subtract(shape, margin)/2.0)
    indices = np.arange(0, count, 1) - np.round(count/2.0)
    pos = np.array([np.add(center_pos, np.multiply(i, separation)) for i in indices])

    return pos

def gen_nonoverlapping_locations(shape, count, separation, margin=0):
    """ Generates `count` number of positions within `shape`, that have minimum
    distance `separation` from each other. The number of positions returned may
    be lower than `count`, because positions too close to each other will be
    deleted. If a `margin` is given, positions will be inside this margin.
    Margin may be tuple-valued.
    """
    positions = gen_random_locations(shape, count, margin)
    return drop_close(positions, separation)


def draw_spots(shape, positions, size, noise_level=0, bitdepth=8, **kwargs):
    """ Generates an image with features at given positions. A feature with
    position x will be centered around pixel x. In other words, the origin of
    the output image is located at the center of pixel (0, 0).

    Parameters
    ----------
    shape : tuple of int
        the shape of the produced image
    positions : iterable of tuples
        an iterable of positions
    size : number
        the size of the feature (meaning depends on feature, for feat_gauss,
        it is the radius of gyration)
    noise_level : int, default: 0
        white noise will be generated up to this level
    bitdepth : int, default: 8
        the desired bitdepth of the image (<=32 bits)
    kwargs : keyword arguments are passed to draw_feature

    See also
    --------
    draw_feature
    """
    if bitdepth <= 8:
        dtype = np.uint8
        internaldtype = np.uint16
    elif bitdepth <= 16:
        dtype = np.uint16
        internaldtype = np.uint32
    elif bitdepth <= 32:
        dtype = np.uint32
        internaldtype = np.uint64
    else:
        raise ValueError('Bitdepth should be <= 32')
    np.random.seed(0)
    image = np.zeros([int(s) for s in shape], dtype=internaldtype)
    if noise_level > 0:
        image += np.random.randint(0, noise_level + 1,
                                   shape).astype(internaldtype)
    for pos in positions:
        draw_feature(image, pos, size, max_value=2**bitdepth - 1, **kwargs)
    return image.clip(0, 2**bitdepth - 1).astype(dtype)


def draw_array(N, size, separation=None, ndim=2, **kwargs):
    """ Generates an image with an array of features. Each feature has a random
    offset of +- 0.5 pixel.

    Parameters
    ----------
    N : int
        the number of features
    size : number
        the size of the feature (meaning depends on feature, for feat_gauss,
        it is the radius of gyration)
    separation : number or tuple
        the desired separation between features
    kwargs : keyword arguments are passed to draw_spots

    See also
    --------
    draw_spots
    draw_feature
    """
    size = validate_tuple(size, ndim)
    if separation is None:
        separation = tuple([sz * 8 for sz in size])
    margin = separation
    Nsqrt = int(N**(1/ndim) + 0.9999)
    pos = np.meshgrid(*[np.arange(0, s * Nsqrt, s) for s in separation],
                      indexing='ij')
    pos = np.array([p.ravel() for p in pos], dtype=float).T[:N] + margin
    pos += (np.random.random(pos.shape) - 0.5)  #randomize subpixel location
    shape = tuple(np.max(pos, axis=0).astype(int) + margin)
    return pos, draw_spots(shape, pos, size, **kwargs)


def rot_2d(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], float)


def rot_3d(angles):
    # Tait-Bryan angles in ZYX convention
    if not hasattr(angles, '__iter__'):
        angles = (angles, 0, 0)
    if len(angles) == 2:
        angles = (angles[0], angles[1], 0)
    s1, s2, s3 = [np.sin(x) for x in angles]
    c1, c2, c3 = [np.cos(x) for x in angles]
    return np.array([[c1*c2, c1*s2*s3-c3*s1, s1*s3+c1*c3*s2],
                     [c2*s1, c1*c3+s1*s2*s3, c3*s1*s2 - c1*s3],
                     [-s2, c2*s3, c2*c3]], float)


clusters_2d = {1: np.array([[0, 0]], float),
               2: np.array([[0, -1], [0, 1]], float),
               3: np.array([[0, 1],
                            [-0.5 * np.sqrt(3), -0.5],
                            [0.5 * np.sqrt(3), -0.5]], float)*2/3*np.sqrt(3),
               4: np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], float)}
clusters_3d = {1: np.array([[0, 0, 0]], float),
               2: np.array([[0, 0, -1], [0, 0, 1]], float),
               3: np.array([[0, 0, 2/np.sqrt(3)],
                            [-1, 0, -1/np.sqrt(3)],
                            [1, 0,  -1/np.sqrt(3)]], float),
               4: np.array([[0, 0, (1/2)*np.sqrt(6)],
                            [0, -(2/3)*np.sqrt(3), -(1/6)*np.sqrt(6)],
                            [1, (1/3)*np.sqrt(3), -(1/6)*np.sqrt(6)],
                            [-1, (1/3)*np.sqrt(3), -(1/6)*np.sqrt(6)]], float)}


def draw_cluster(image, position, size, cluster_size, hard_radius=1., angle=0,
                 **kwargs):
    """Draws a cluster of size `n` at `pos` with angle `angle`. The distance
    between particles is determined by `hard_radius`."""
    if image.ndim == 2:
        rot = rot_2d(angle)
        coord = clusters_2d[cluster_size]
    elif image.ndim == 3:
        rot = rot_3d(angle)
        coord = clusters_3d[cluster_size]
    coord = np.dot(coord, rot.T)  # rotate
    coord *= hard_radius * np.array(size)[np.newaxis, :]  # scale
    coord += np.array(position)[np.newaxis, :]  # translate
    for pos in coord:
        draw_feature(image, pos, size, **kwargs)
    return coord

class SimulatedImage:
    """ This class makes it easy to generate artificial pictures.

    Parameters
    ----------
    shape : tuple of int
    dtype : numpy.dtype, default np.uint8
    saturation : maximum value in image
    hard_radius : default radius of particles, used for determining the
                  distance between particles in clusters
    feat_dict : dictionary of arguments passed to tp.artificial.draw_feature

    Attributes
    ----------
    image : ndarray containing pixel values
    center : the center [y, x] to use for radial coordinates

    Examples
    --------
    image = SimulatedImage(shape=(50, 50), dtype=np.uint8, hard_radius=7,
                           feat_dict={'diameter': 20, 'max_value': 100,
                                      'feat_func': SimulatedImage.feat_hat,
                                      'disc_size': 0.2})
    image.draw_feature((10, 10))
    image.draw_dimer((32, 35), angle=75)
    image.add_noise(5)
    image()
    """
    def __init__(self, shape, size, dtype=np.uint8, saturation=None,
                 hard_radius=None, signal=None, noise=0,
                 feat_func=feat_gauss, **feat_kwargs):
        self.ndim = len(shape)
        self.shape = shape
        self.dtype = dtype
        self.image = np.zeros(shape, dtype=dtype)
        if _Frame is not None:
            self.image = _Frame(self.image)
        self.size = validate_tuple(size, self.ndim)
        self.isotropic = np.all([self.size[1:] == self.size[:-1]])
        self.feat_func = feat_func
        self.feat_kwargs = feat_kwargs
        self.noise = noise
        if saturation is None and np.issubdtype(dtype, np.integer):
            self.saturation = np.iinfo(dtype).max
        elif saturation is None and np.issubdtype(dtype, np.floating):
            self.saturation = 1
        else:
            self.saturation = saturation
        if signal is None:
            self.signal = self.saturation
        else:
            self.signal = signal
        self.center = tuple([s // 2 for s in shape])
        self.hard_radius = hard_radius
        self._coords = []
        self.pos_columns = ['z', 'y', 'x'][-self.ndim:]
        if self.isotropic:
            self.size_columns = ['size']
        else:
            self.size_columns = ['size_z', 'size_y', 'size_x'][-self.ndim:]

    def __call__(self):
        # so that you can checkout the image with image() instead of image.image
        return self.noisy_image(self.noise)

    def clear(self):
        """Clears the current image"""
        self._coords = []
        self.image = np.zeros_like(self.image)

    def draw_feature(self, pos):
        """Draws a feature at `pos`."""
        pos = [float(p) for p in pos]
        self._coords.append(pos)
        draw_feature(self.image, pos, self.size, self.signal,
                     self.feat_func, **self.feat_kwargs)

    def draw_features(self, N, separation=0, margin=None):
        """Draws N features at random locations, using minimum separation
        and a margin. If separation > 0, less than N features may be drawn."""
        if margin is None:
            margin = self.hard_radius
        if margin is None:
            margin = 0
        pos = gen_random_locations(self.shape, N, margin)
        if separation > 0:
            pos = drop_close(pos, separation)
        for p in pos:
            self.draw_feature(p)
        return pos

    def draw_feature_radial(self, r, angle, center=None):
        """Draws a feature at radial coordinates `r`, `angle`. The center
        of the radial coordinates system is determined by `center`. If this
        is not given, self.center is used.

        For 3D, angle has to be a tuple of length 2: (phi, theta), in which
        theta is the angle with the positive z axis."""
        if center is None:
            center = self.center
        if self.ndim == 2:
            pos = (center[0] + self.size[0]*r*np.sin(angle*(np.pi/180)),
                   center[1] + self.size[1]*r*np.cos(angle*(np.pi/180)))
        elif self.ndim == 3:
            if not hasattr(angle, '__iter__'):
                angle = (angle, 0)
            r_sin_theta = r*np.sin(angle[1]*(np.pi/180))
            pos = (center[0] + self.size[0]*r*np.cos(angle[1]*(np.pi/180)),
                   center[1] + self.size[1]*r_sin_theta*np.sin(angle[0]*(np.pi/180)),
                   center[2] + self.size[2]*r_sin_theta*np.cos(angle[0]*(np.pi/180)))
        else:
            raise ValueError("Don't know how to draw in {} dimensions".format(self.ndim))
        self.draw_feature(pos)
        return pos

    def draw_dimer(self, pos, angle, hard_radius=None):
        """Draws a dimer at `pos` with angle `angle`. The distance
        between particles is determined by 2*`hard_radius`. If this is not
        given, self.separation is used."""
        return self.draw_cluster(2, pos, angle, hard_radius)
    draw_dumbell = draw_dimer

    def draw_trimer(self, pos, angle, hard_radius=None):
        """Draws a trimer at `pos` with angle `angle`. The distance
        between particles is determined by `separation`. If this is not
        given, self.separation is used."""
        return self.draw_cluster(3, pos, angle, hard_radius)
    draw_triangle = draw_trimer

    def draw_cluster(self, cluster_size, center=None, angle=0, hard_radius=None):
        """Draws a cluster of size `n` at `pos` with angle `angle`. The distance
        between particles is determined by `separation`. If this is not
        given, self.separation is used."""
        if hard_radius is None:
            hard_radius = self.hard_radius
        if center is None:
            center = self.center
        if self.ndim == 2:
            rot = rot_2d(angle)
            coord = clusters_2d[cluster_size]
        elif self.ndim == 3:
            rot = rot_3d(angle)
            coord = clusters_3d[cluster_size]
        coord = np.dot(coord, rot.T)  # rotate
        coord *= hard_radius * np.array(self.size)[np.newaxis, :]  # scale
        coord += np.array(center)[np.newaxis, :]  # translate
        for pos in coord:
            self.draw_feature(pos)
        return coord

    def draw_clusters(self, N, cluster_size, hard_radius=None, separation=0,
                      margin=None):
        """Draws N clusters at random locations, using minimum separation
        and a margin. If separation > 0, less than N features may be drawn."""
        if hard_radius is None:
            hard_radius = self.hard_radius
        if margin is None:
            margin = self.hard_radius
        if margin is None:
            margin = 0
        pos = gen_random_locations(self.shape, N, margin)
        if separation > 0:
            pos = drop_close(pos, separation)

        if self.ndim == 2:
            angles = np.random.uniform(0, 2*np.pi, N)
        elif self.ndim == 3:
            angles = np.random.uniform(0, 2*np.pi, (N, 3))

        for p, a in zip(pos, angles):
            self.draw_cluster(cluster_size, p, a, hard_radius)
        return pos

    def noisy_image(self, noise_level):
        """Adds noise to the current image, uniformly distributed
        between 0 and `noise_level`, not including noise_level."""
        if noise_level <= 0:
            return self.image
        if np.issubdtype(self.dtype, np.integer):
            noise = np.random.poisson(noise_level, self.shape)
        else:
            noise = np.clip(np.random.normal(noise_level, noise_level/2, self.shape), 0, self.saturation)
        noisy_image = np.clip(self.image + noise, 0, self.saturation)
        result = np.array(noisy_image, dtype=self.dtype)
        if _Frame is not None:
            result = _Frame(result)
        return result

    def denoised(self, noise_level, noise_size, smoothing_size=None,
                 threshold=None):
        image = self.noisy_image(noise_level)
        return bandpass(image, noise_size, smoothing_size, threshold)

    @property
    def coords(self):
        if len(self._coords) == 0:
            return np.zeros((0, self.ndim), dtype=float)
        return np.array(self._coords)

    def f(self, noise=0):
        result = self.coords + np.random.random(self.coords.shape) * noise
        result = pd.DataFrame(result, columns=self.pos_columns)
        result['signal'] = float(self.signal)
        if self.isotropic:
            result[self.size_columns[0]] = float(self.size[0])
        else:
            for col, s in zip(self.size_columns, self.size):
                result[col] = float(s)
        return result

def feat_brightfield(r, ndim, radius, dark_value, bright_value, dip):
    """ Brightfield particle with intensity dip in center at r = 0. """
    image = np.zeros_like(r)

    # by definition
    r_rel = 1.0
    r_factor = radius / r_rel
    thickness = r_rel*0.1

    if thickness*r_factor < 2.0:
        thickness = 2.0 / r_factor

    mask = r < r_rel
    mask_ring = mask & (r > (r_rel-thickness))

    if dip:
        mask_dip = r < thickness
        image[mask_dip] += 1.5*dark_value

    gauss_radius = r_rel-1*thickness
    image[mask] += bright_value*np.exp(-(r[mask]/(gauss_radius**2))**2)

    image[mask_ring] = dark_value

    return image

def draw_features_brightfield(shape, positions, radius, noise_level=0,
                              bitdepth=8, background=0.5, dip=False, **kwargs):
    """ Generates an image with features at given positions. A feature with
    position x will be centered around pixel x. In other words, the origin of
    the output image is located at the center of pixel (0, 0).

    Parameters
    ----------
    shape : tuple of int
        the shape of the produced image
    positions : iterable of tuples
        an iterable of positions
    radius : number
        the radius of the feature
    noise_level : int, default: 0
        white noise will be generated up to this level
    bitdepth : int, default: 8
        the desired bitdepth of the image (<=32 bits)
    background : float, default: 0.5
        the value of the background ranging from 0 (black) to 1 (white)
    kwargs : keyword arguments are passed to draw_feature

    See also
    --------
    draw_feature_brightfield
    """
    if bitdepth <= 8:
        dtype = np.uint8
        internaldtype = np.uint16
    elif bitdepth <= 16:
        dtype = np.uint16
        internaldtype = np.uint32
    elif bitdepth <= 32:
        dtype = np.uint32
        internaldtype = np.uint64
    else:
        raise ValueError('Bitdepth should be <= 32')
    np.random.seed(0)

    max_brightness = 2**bitdepth - 1

    if background is None or background < 0 or background > 1:
        background = 0.5

    image = background*max_brightness*np.ones([int(s) for s in shape], 
                                              dtype=internaldtype)

    if noise_level > 0:
        image += np.random.randint(0, noise_level + 1,
                                   shape).astype(internaldtype)

    if np.max(image) > max_brightness:
        image = image/np.max(image)*max_brightness

    kwargs['feat_func'] = feat_brightfield
    kwargs['dark_value'] = -0.3
    kwargs['bright_value'] = 0.8
    kwargs['dip'] = dip
    kwargs['radius'] = radius[0]

    for pos in positions:
        draw_feature(image, pos, radius, max_value=max_brightness, **kwargs)
    result = image.clip(0, 2**bitdepth - 1).astype(dtype)

    return result

