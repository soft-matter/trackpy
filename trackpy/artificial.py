from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np
import pandas as pd
from pims import Frame
from scipy.spatial import cKDTree
from trackpy.utils import validate_tuple


def draw_point(image, pos, value):
    image[tuple(pos)] = value


def feat_gauss(r, rg=0.333):
    """ Gaussian at r = 0 with max value of 1. Its radius of gyration is
    given by rg. """
    return np.exp((r/rg)**2 * r.ndim/-2)


def feat_gauss_edge(r, value_at_edge=0.1):
    """ Gaussian at r = 0 with max value of 1. Its value at r = 1 is given by
    value_at_edge. """
    return np.exp(np.log(value_at_edge)*r**2)


def feat_ring(r, r_at_max, value_at_edge=0.1):
    """ Ring feature with a gaussian profile, centered at r_at_max. Its value
    at r = 1 is given by value_at_edge."""
    return np.exp(np.log(value_at_edge)*((r - r_at_max) / (1 - r_at_max))**2)


def feat_hat(r, disc_size, value_at_edge=0.1):
    """ Solid disc of size disc_size, with Gaussian smoothed borders. """
    mask = r > disc_size
    spot = (~mask).astype(r.dtype)
    spot[mask] = feat_ring(r[mask], disc_size, value_at_edge)
    spot[~mask] = 1
    return spot


def feat_step(r):
    """ Solid disc. """
    return r <= 1


class SimulatedImage(object):
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
        self.image = Frame(np.zeros(shape, dtype=dtype))
        self.size = validate_tuple(size, self.ndim)
        self.isotropic = np.all([self.size[1:] == self.size[:-1]])
        self.feat_func = feat_func
        self.feat_kwargs = feat_kwargs
        self.feat_kwargs['rg'] = 0.25
        self.noise = noise
        if saturation is None and np.issubdtype(dtype, np.integer):
            self.saturation = np.iinfo(dtype).max
        elif saturation is None and np.issubdtype(dtype, np.float):
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
        draw_feature(self.image, pos, [s*8 for s in self.size], self.signal,
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
            pos = eliminate_overlapping_locations(pos, separation)
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
                   center[1] + self.size[1] * r_sin_theta * np.sin(angle[0]*(np.pi/180)),
                   center[2] + self.size[2] * r_sin_theta * np.cos(angle[0]*(np.pi/180)))
        else:
            raise ValueError("Don't know how to draw in {} dimensions".format(self.ndim))
        self.draw_feature(pos)
        return pos

    def draw_dimer(self, pos, angle, hard_radius=None):
        """Draws a dimer at `pos` with angle `angle`. The distance
        between particles is determined by 2*`hard_radius`. If this is not
        given, self.separation is used."""
        if hard_radius is None:
            hard_radius = self.hard_radius
        if self.ndim == 2:
            angles = [angle, angle + 180]
        if self.ndim == 3:
            if not hasattr(angle, '__iter__'):
                angle = (angle, 0)
            angles = [angle, (angle[0] + 180, 180 - angle[1])]
        res = [self.draw_feature_radial(hard_radius, a, pos) for a in angles]
        return res
    draw_dumbell = draw_dimer

    def draw_trimer(self, pos, angle, hard_radius=None):
        """Draws a trimer at `pos` with angle `angle`. The distance
        between particles is determined by `separation`. If this is not
        given, self.separation is used."""
        if hard_radius is None:
            hard_radius = self.hard_radius
        d = hard_radius*2/3*np.sqrt(3)
        y1, x1 = self.draw_feature_radial(d, angle, pos)
        y2, x2 = self.draw_feature_radial(d, angle+120, pos)
        y3, x3 = self.draw_feature_radial(d, angle+240, pos)
        return [y1, x1], [y2, x2], [y3, x3]
    draw_triangle = draw_trimer

    def noisy_image(self, noise_level):
        """Adds noise to the current image, uniformly distributed
        between 0 and `noise_level`, not including noise_level."""
        if noise_level <= 0:
            return self.image
        if np.issubdtype(self.dtype, np.integer):
            noise = np.random.randint(0, noise_level, self.shape)
        else:
            noise = np.random.random(self.shape) * noise_level
        noisy_image = np.clip(self.image + noise, 0, self.saturation)
        return Frame(np.array(noisy_image, dtype=self.dtype))

    @property
    def coords(self):
        if len(self._coords) == 0:
            return np.zeros((0, self.ndim), dtype=np.float)
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


def draw_feature(image, position, diameter, max_value=None,
                 feat_func=feat_gauss, ecc=None, **kwargs):
    """ Draws a radial symmetric feature and adds it to the image at given
    position. The given function will be evaluated at each pixel coordinate,
    no averaging or convolution is done.

    Parameters
    ----------
    image : ndarray
        image to draw features on
    position : iterable
        coordinates of feature position
    diameter : number
        defines the box that will be drawn on
    max_value : number
        maximum feature value. should be much less than the max value of the
        image dtype, to avoid pixel wrapping at overlapping features
    feat_func : function. Default: feat_gauss
        function f(r) that takes an ndarray of radius values
        and returns intensity values <= 1
    ecc : positive number, optional
        eccentricity of feature, defined only in 2D. Identical to setting
        diameter to (diameter / (1 - ecc), diameter * (1 - ecc))
    kwargs : keyword arguments are passed to feat_func
    """
    if len(position) != image.ndim:
        raise ValueError("Number of position coordinates should match image"
                         " dimensionality.")
    diameter = validate_tuple(diameter, image.ndim)
    if ecc is not None:
        if len(diameter) != 2:
            raise ValueError("Eccentricity is only defined in 2 dimensions")
        if diameter[0] != diameter[1]:
            raise ValueError("Diameter is already anisotropic; eccentricity is"
                             " not defined.")
        diameter = (diameter[0] / (1 - ecc), diameter[1] * (1 - ecc))
    radius = tuple([d / 2 for d in diameter])
    if max_value is None:
        max_value = np.iinfo(image.dtype).max - 3
    rect = []
    vectors = []
    for (c, r, lim) in zip(position, radius, image.shape):
        if (c >= lim) or (c < 0):
            raise ValueError("Position outside of image.")
        lower_bound = max(int(np.floor(c - r)), 0)
        upper_bound = min(int(np.ceil(c + r + 1)), lim)
        rect.append(slice(lower_bound, upper_bound))
        vectors.append(np.arange(lower_bound - c, upper_bound - c) / r)
    coords = np.meshgrid(*vectors, indexing='ij', sparse=True)
    r = np.sqrt(np.sum(np.array(coords)**2, axis=0))
    spot = max_value * feat_func(r, **kwargs)
    image[rect] += spot.astype(image.dtype)


def gen_random_locations(shape, count, margin=0):
    """ Generates `count` number of positions within `shape`. If a `margin` is
    given, positions will be inside this margin. Margin may be tuple-valued.
    """
    margin = validate_tuple(margin, len(shape))
    np.random.seed(0)
    pos = [np.random.randint(round(m), round(s - m), count)
           for (s, m) in zip(shape, margin)]
    return np.array(pos).T


def eliminate_overlapping_locations(f, separation):
    """ Makes sure that no position is within `separation` from each other, by
    deleting one of the that are to close to each other.
    """
    separation = validate_tuple(separation, f.shape[1])
    assert np.greater(separation, 0).all()
    # Rescale positions, so that pairs are identified below a distance of 1.
    f = f / separation
    while True:
        duplicates = cKDTree(f, 30).query_pairs(1)
        if len(duplicates) == 0:
            break
        to_drop = []
        for pair in duplicates:
            to_drop.append(pair[1])
        f = np.delete(f, to_drop, 0)
    return f * separation


def gen_nonoverlapping_locations(shape, count, separation, margin=0):
    """ Generates `count` number of positions within `shape`, that have minimum
    distance `separation` from each other. The number of positions returned may
    be lower than `count`, because positions too close to each other will be
    deleted. If a `margin` is given, positions will be inside this margin.
    Margin may be tuple-valued.
    """
    positions = gen_random_locations(shape, count, margin)
    return eliminate_overlapping_locations(positions, separation)


def draw_spots(shape, positions, diameter, noise_level=0, bitdepth=8,
               feat_func=feat_gauss, ecc=None, **kwargs):
    """ Generates an image with features at given positions. A feature with
    position x will be centered around pixel x. In other words, the origin of
    the output image is located at the center of pixel (0, 0).

    Parameters
    ----------
    shape : tuple of int
        the shape of the produced image
    positions : iterable of tuples
        an iterable of positions
    diameter : number or tuple
        the sizes of the box that will be used per feature. The actual feature
        'size' is determined by feat_func and kwargs given to feat_func.
    noise_level : int, default: 0
        white noise will be generated up to this level
    bitdepth : int, default: 8
        the desired bitdepth of the image (<=32 bits)
    feat_func : function, default: feat_gauss
        function f(r) that takes an ndarray of radius values
        and returns intensity values <= 1
    ecc : positive number, optional
        eccentricity of feature, defined only in 2D. Identical to setting
        diameter to (diameter / (1 - ecc), diameter * (1 - ecc))
    kwargs : keyword arguments are passed to feat_func
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
    image = np.random.randint(0, noise_level + 1, shape).astype(internaldtype)
    for pos in positions:
        draw_feature(image, pos, diameter, max_value=2**bitdepth - 1,
                     feat_func=feat_func, ecc=ecc, **kwargs)
    return image.clip(0, 2**bitdepth - 1).astype(dtype)
