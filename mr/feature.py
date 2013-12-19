# Copyright 2012 Daniel B. Allan
# dallan@pha.jhu.edu, daniel.b.allan@gmail.com
# http://pha.jhu.edu/~dallan
# http://www.danallan.com
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses>.

from __future__ import division
from scipy import ndimage
from scipy import stats
import logging
import warnings
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from mr import uncertainty
from mr.preprocessing import bandpass, scale_to_gamut
from C_fallback_python import nullify_secondary_maxima
from mr.utils import memo
from .print_update import print_update


logger = logging.getLogger(__name__)


def local_maxima(image, radius, separation, percentile=64):
    """Find local maxima whose brightness is above a given percentile."""

    # Compute a threshold based on percentile.
    not_black = image[np.nonzero(image)]
    threshold = stats.scoreatpercentile(not_black, percentile)
    ndim = image.ndim

    # The intersection of the image with its dilation gives local maxima.
    if not np.issubdtype(image.dtype, np.integer):
        raise TypeError("Perform dilation on exact (i.e., integer) data.")
    footprint = binary_mask(radius, ndim, separation)
    dilation = ndimage.grey_dilation(image, footprint=footprint,
                                     mode='constant')
    maxima = np.where((image == dilation) & (image > threshold))
    if not np.size(maxima) > 0:
        _warn_no_maxima()
        return np.empty((0, ndim))

    # Flat peaks, for example, return multiple maxima. Eliminate them.
    maxima_map = np.zeros_like(image)
    maxima_map[maxima] = image[maxima]
    footprint = binary_mask(separation, ndim, separation)
    maxima_map = ndimage.generic_filter(
        maxima_map, nullify_secondary_maxima(), footprint=footprint,
        mode='constant')
    maxima = np.where(maxima_map > 0)

    # Do not accept peaks near the edges.
    margin = int(separation)//2
    maxima_map[..., -margin:] = 0
    maxima_map[..., :margin] = 0
    if ndim > 1:
        maxima_map[..., -margin:, :] = 0
        maxima_map[..., :margin, :] = 0
    if ndim > 2:
        maxima_map[..., -margin:, :, :] = 0
        maxima_map[..., :margin, :, :] = 0
    if ndim > 3:
        raise NotImplementedError("I tap out beyond three dimensions.")
        # TODO Change if into loop using slice(None) as :
    maxima = np.where(maxima_map > 0)
    if not np.size(maxima) > 0:
        warnings.warn("Bad image! All maxima were in the margins.",
                      UserWarning)

    # Return coords in as a numpy array shaped so it can be passed directly
    # to the DataFrame constructor.
    return np.vstack(maxima).T


def estimate_mass(image, radius, coord):
    "Compute the total brightness in the neighborhood of a local maximum."
    square = [slice(c - radius, c + radius + 1) for c in coord]
    neighborhood = binary_mask(radius, image.ndim)*image[square]
    return np.sum(neighborhood)


def estimate_size(image, radius, coord, estimated_mass):
    "Compute the total brightness in the neighborhood of a local maximum."
    square = [slice(c - radius, c + radius + 1) for c in coord]
    neighborhood = binary_mask(radius, image.ndim)*image[square]
    Rg = np.sqrt(np.sum(r_squared_mask(radius, image.ndim)*neighborhood)/
                 estimated_mass)
    return Rg

# center_of_mass can have divide-by-zero errors, avoided thus:
_safe_center_of_mass = lambda x: np.array(ndimage.center_of_mass(x + 1))


def refine(raw_image, image, radius, coord, iterations=10):
    """Characterize the neighborhood of a local maximum, and iteratively
    hone in on its center-of-brightness. Return its coordinates, integrated
    brightness, size (Rg), and eccentricity (0=circular)."""

    ndim = image.ndim
    mask = binary_mask(radius, ndim)

    # Define the circular neighborhood of (x, y).
    square = [slice(c - radius, c + radius + 1) for c in coord]
    neighborhood = mask*image[square]
    cm_n = _safe_center_of_mass(neighborhood)  # neighborhood coords
    cm_i = cm_n - radius + coord  # image coords
    allow_moves = True
    for iteration in range(iterations):
        off_center = cm_n - radius
        if np.all(np.abs(off_center) < 0.005):
            break  # Accurate enough.

        # If we're off by more than half a pixel in any direction, move.
        elif np.any(np.abs(off_center) > 0.6) and allow_moves:
            new_coord = coord
            new_coord[off_center > 0.6] -= 1
            new_coord[off_center < -0.6] += 1
            # Don't move outside the image!
            upper_bound = np.array(image.shape) - 1 - radius
            new_coord = np.clip(new_coord, radius, upper_bound)
            square = [slice(c - radius, c + radius + 1) for c in new_coord]
            neighborhood = mask*image[square]

        # If we're off by less than half a pixel, interpolate.
        else:
            # second-order spline.
            neighborhood = ndimage.shift(neighborhood, -off_center, order=2,
                                         mode='constant', cval=0)
            new_coord = coord + off_center
            # Disallow any whole-pixels moves on future iterations.
            allow_moves = False

        cm_n = _safe_center_of_mass(neighborhood)  # neighborhood coords
        cm_i = cm_n - radius + new_coord  # image coords
        coord = new_coord

    # Characterize the neighborhood of our final centroid.
    mass = neighborhood.sum()
    Rg = np.sqrt(np.sum(r_squared_mask(radius, ndim)*neighborhood)/mass)
    # I only know how to measure eccentricity in 2D.
    if ndim == 2:
        ecc = np.sqrt(np.sum(neighborhood*cosmask(radius))**2 +
                      np.sum(neighborhood*sinmask(radius))**2)
        ecc /= (mass - neighborhood[radius, radius] + 1e-6)
    else:
        ecc = np.nan
    raw_neighborhood = mask*raw_image[square]
    signal = raw_neighborhood.max()  # black_level subtracted later

    # matplotlib and ndimage have opposite conventions for xy <-> yx.
    final_coords = cm_i[..., ::-1]
    return np.array(list(final_coords) + [mass, Rg, ecc, signal])


def locate(image, diameter, minmass=100., maxsize=None, separation=None,
           noise_size=1, smoothing_size=None, threshold=1, invert=False,
           percentile=64, topn=None, preprocess=True):
    """Read an image, do optional image preparation and cleanup, and locate
    Gaussian-like blobs of a given size above a given total brightness.

    Parameters
    ----------
    image: image array
    diameter : feature size in px
    minmass : minimum integrated brightness
       Default is 100, but a good value is often much higher. This is a
       crucial parameter for elminating spurrious features.
    maxsize : maximum radius-of-gyration of brightness, default None
    separation : feature separation in px
    noise_size : scale of Gaussian blurring. Default 1.
    smoothing_size : defauls to separation
    threshold : Clip bandpass result below this value.
        Default 1; use 8 for 16-bit images.
    invert : Set to True if features are darker than background. False by
        default.
    percentile : Features must have a peak brighter than pixels in this
        percentile. This helps eliminate spurrious peaks.
    topn : Return only the N brightest features above minmass. 
        If None (default), return all features above minmass.
    preprocess : Set to False to turn out automatic preprocessing.

    Returns
    -------
    DataFrame([x, y, mass, size, ecc, signal])

    where mass means total integrated brightness of the blob
    and size means the radius of gyration of its Gaussian-like profile
    """

    # Validate parameters and set defaults.
    if not diameter & 1:
        raise ValueError("Feature diameter must be an odd number. Round up.")
    if not separation:
        separation = int(diameter) + 1
    radius = int(diameter)//2
    if smoothing_size is None:
        smoothing_size = diameter
    image = np.squeeze(image)
    if preprocess:
        if invert:
            # It is tempting to do this in place, but if it is called multiple
            # times on the same image, chaos reigns.
            max_value = np.iinfo(image.dtype).max
            image = image ^ max_value
        bp_image = bandpass(image, noise_size, smoothing_size, threshold)
    else:
        bp_image = image.copy()
    bp_image = scale_to_gamut(bp_image, image.dtype)

    # Find local maxima.
    coords = local_maxima(bp_image, radius, separation, percentile)
    count_maxima = coords.shape[0]

    approx_mass = np.empty(count_maxima)  # initialize to avoid appending
    for i in range(count_maxima):
        approx_mass[i] = estimate_mass(bp_image, radius, coords[i])
    if maxsize is not None:
        approx_size = np.empty(count_maxima)
        for i in range(count_maxima):
            approx_size[i] = estimate_size(bp_image, radius, coords[i], 
                                           approx_mass[i])
        coords = coords[(approx_mass > minmass) & (approx_size < maxsize)]
    else:
        coords = coords[approx_mass > minmass]
    count_qualified = coords.shape[0]

    # Refine their locations and characterize mass, size, etc.
    ndim = image.ndim
    refined_coords = np.empty((count_qualified, ndim + 4))
    for i in range(count_qualified):
        refined_coords[i] = refine(image, bp_image, radius, coords[i])

    # Filter by minmass again, using final ("exact") mass.
    exact_mass = refined_coords[:, ndim]
    refined_coords = refined_coords[exact_mass > minmass]
    count_qualified = refined_coords.shape[0]

    if topn is not None and count_qualified > topn:
        exact_mass = exact_mass[exact_mass > minmass]
        if topn == 1:
            # special case for high performance and correct shape
            refined_coords = refined_coords[np.argmax(exact_mass)]
            refined_coords = refined_coords.reshape(1, -1)
        else:
            refined_coords = refined_coords[np.argsort(exact_mass)][-topn:]

    # Present the results in a DataFrame.
    logger.info("%s local maxima, %s of qualifying mass",
                count_maxima, count_qualified)
    if ndim < 4:
        coord_columns = ['x', 'y', 'z'][:image.ndim]
    else:
        coord_columns = map(lambda i: 'x' + str(i), range(ndim))
    columns = coord_columns + ['mass', 'size', 'ecc', 'signal']
    if len(refined_coords) == 0:
        return DataFrame(columns=columns)  # TODO fill with np.empty
    f = DataFrame(refined_coords, columns=columns)
    black_level, noise = uncertainty.measure_noise(image, diameter, threshold)
    f['signal'] -= black_level
    ep = uncertainty.static_error(f, noise, diameter, noise_size)
    f = f.join(ep)
    return f


def batch(frames, diameter, minmass=100, maxsize=None, separation=None,
          noise_size=1, smoothing_size=None, threshold=1, invert=False,
          percentile=64, topn=None, preprocess=True,
          store=None, conn=None, sql_flavor=None, table=None,
          do_not_return=False):
    """Process a list of images, doing optional image preparation and cleanup,
    locating Gaussian-like blobs of a given size.

    Parameters
    ----------
    frames : iterable frames
        For example, frames = mr.video.frame_generator('video_file.avi')
                  or frames = [array1, array2, array3]
    diameter : feature size in px
    minmass : minimum integrated brightness
       Default is 100, but a good value is often much higher. This is a
       crucial parameter for elminating spurrious features.
    maxsize : maximum radius-of-gyration of brightness, default None
    separation : feature separation in px. Default = 1 + diamter.
    noise_size : scale of Gaussian blurring. Default = 1.
    smoothing_size : Default = separation.
    threshold : Clip bandpass result below this value.
        Default 1; use 8 for 16-bit images.
    invert : Set to True if features are darker than background. False by
        default.
    percentile : Features must have a peak brighter than pixels in this
        percentile. This helps eliminate spurrious peaks.
    topn : Return only the N brightest features above minmass. 
        If None (default), return all features above minmass.
    preprocess : Set to False to turn out automatic preprocessing.
    store : Optional HDFStore
    conn : Optional connection to a SQL database
    sql_flavor : If using a SQL connection, specify 'sqlite' or 'MySQL'.
    table : If using HDFStore or SQL, specify table name.
        Default: 'features_timestamp'.
    do_not_return : Save the result frame by frame, but do not return it when
        finished. Conserved memory for parallel jobs.

    Returns
    -------
    DataFrame([x, y, mass, size, ecc, signal])

    where mass means total integrated brightness of the blob
    and size means the radius of gyration of its Gaussian-like profile
    """
    timestamp = pd.datetime.utcnow().strftime('%Y-%m-%d-%H%M%S')
    if table is None:
        table = 'features-' + timestamp
    # Gather meta information and pack it into a Series.
    try:
        source = frames.filename
    except:
        source = None
    meta = Series([source, diameter, minmass, separation, noise_size,
                   smoothing_size, invert, percentile, topn,
                   pd.Timestamp(timestamp)],
                  index=['source',
                         'diameter', 'minmass', 'separation', 'noise_size',
                         'smoothing_size', 'invert', 'percentile', 'topn',
                         'timestamp'])
    all_centroids = []
    for i, image in enumerate(frames):
        # If frames has a cursor property, use it. Otherwise, just count
        # the frames from 0.
        try:
            frame_no = frames.cursor - 1
        except AttributeError:
            frame_no = i
        centroids = locate(image, diameter, minmass, maxsize, separation,
                           noise_size, smoothing_size, threshold, invert,
                           percentile, topn, preprocess)
        centroids['frame'] = frame_no
        message = "Frame %d: %d features" % (frame_no, len(centroids))
        logger.info(message)
        print_update(message)
        if len(centroids) == 0:
            continue
        indexed = ['frame']  # columns on which you can perform queries
        if store is not None:
            store.append(table, centroids, data_columns=indexed)
            store.flush()  # Force save. Not essential.
        elif conn is not None:
            if sql_flavor is None:
                raise ValueError("Specifiy sql_flavor: MySQL or sqlite.")
            pd.io.sql.write_frame(centroids, table, conn,
                                  flavor=sql_flavor, if_exists='append')
        else:
            all_centroids.append(centroids)
    if do_not_return:
        return None
    if store is not None:
        store.get_storer(table).attrs.meta = meta
        return store[table]
    elif conn is not None:
        return pd.io.sql.read_frame("SELECT * FROM %s" % table, conn)
    else:
        return pd.concat(all_centroids).reset_index(drop=True)


@memo
def binary_mask(radius, ndim, separation=None):
    "circular mask in a square array"
    points = np.arange(-radius, radius + 1)
    if ndim > 1:
        coords = np.array(np.meshgrid(*([points]*ndim)))
    else:
        coords = points.reshape(1, -1)
    r = np.sqrt(np.sum(coords**2, 0))
    return r <= radius


@memo
def r_squared_mask(radius, ndim):
    points = np.arange(-radius, radius + 1)
    if ndim > 1:
        coords = np.array(np.meshgrid(*([points]*ndim)))
    else:
        coords = points.reshape(1, -1)
    r2 = np.sum(coords**2, 0)
    r2[r2 > radius**2] = 0
    return r2


@memo
def theta_mask(radius):
    # 2D only
    tan_of_coord = lambda y, x: np.arctan2(radius - y, x - radius)
    diameter = 2*radius + 1
    return np.fromfunction(tan_of_coord, (diameter, diameter))


@memo
def sinmask(radius):
    return np.sin(2*theta_mask(radius))


@memo
def cosmask(radius):
    return np.cos(2*theta_mask(radius))


@memo
def _warn_no_maxima():
    warnings.warn("No local maxima were found.", UserWarning)
