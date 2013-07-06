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
import re
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series
from scipy.ndimage import morphology
from scipy.ndimage import filters
from scipy.ndimage import fourier
from scipy.ndimage import measurements
from scipy.ndimage import interpolation
from scipy import stats
from mr.core.utils import memo
from mr.core import uncertainty
from mr.core import plots
from mr.core.preprocessing import (bandpass, circular_mask, rgmask, thetamask,
                                   sinmask, cosmask, scale_to_gamut)
import _Cfilters
import warnings

logger = logging.getLogger(__name__)

def local_maxima(image, diameter, separation, percentile=64):
    """Find local maxima whose brightness is above a given percentile."""
    # Find the threshold brightness, representing the given
    # percentile among all NON-ZERO pixels in the image.
    flat = np.ravel(image) 
    nonblack = flat[flat > 0]
    if len(nonblack) == 0:
        warnings.warn("All pixels are black.")
        return np.empty((0, 2))
    threshold = stats.scoreatpercentile(flat[flat > 0], percentile)
    # The intersection of the image with its dilation gives local maxima.
    assert np.issubdtype(image.dtype, np.integer), \
        "Perform dilation on exact (i.e., integer) data." 
    dilation = morphology.grey_dilation(
        image, footprint=circular_mask(diameter, separation))
    maxima = np.where((image == dilation) & (image > threshold))
    if not np.size(maxima) > 0:
        warnings.warn("Found zero maxima above the {}"
                           "-percentile treshold at {}.".format(
                           percentile, threshold))
        return np.empty((0, 2))
    # Flat peaks, for example, return multiple maxima.
    # Eliminate redundancies within the separation distance.
    maxima_map = np.zeros_like(image)
    maxima_map[maxima] = image[maxima]
    peak_map = filters.generic_filter(
        maxima_map, _Cfilters.nullify_secondary_maxima(), 
        footprint=circular_mask(separation), mode='constant')
    # Also, do not accept peaks near the edges.
    margin = int(separation)//2
    peak_map[..., :margin] = 0
    peak_map[..., -margin:] = 0
    peak_map[:margin, ...] = 0
    peak_map[-margin:, ...] = 0
    peaks = np.where(peak_map != 0)
    if not np.size(peaks) > 0:
        raise ValueError, "Bad image! All maxima were in the margins."
    # Return coords in as a numpy array, shaped so it can be passed directly
    # to the DataFrame constructor.
    return np.array([peaks[1], peaks[0]]).T # columns: x, y

def estimate_mass(image, x, y, diameter):
    "Compute the total brightness in the neighborhood of a local maximum."
    r = int(diameter)//2
    x0 = x - r
    x1 = x + r + 1
    y0 = y - r
    y1 = y + r + 1
    neighborhood = circular_mask(diameter)*image[y0:y1, x0:x1]
    return np.sum(neighborhood)

def refine_centroid(raw_image, bp_image, x, y, diameter, minmass=100, iterations=10):
    """Characterize the neighborhood of a local maximum, and iteratively
    hone in on its center-of-brightness. Return its coordinates, integrated
    brightness, size (Rg), and eccentricity (0=circular)."""
    # Define the square neighborhood of (x, y).
    r = int(diameter)//2
    x0, y0 = x - r, y - r
    x1, y1 = x + r + 1, y + r + 1
    neighborhood = circular_mask(diameter)*bp_image[y0:y1, x0:x1]
    yc, xc = measurements.center_of_mass(neighborhood)  # neighborhood coords
    yc, xc = yc + y0, xc + x0  # image coords
    ybounds = (0, bp_image.shape[0] - 1 - 2*r)
    xbounds = (0, bp_image.shape[1] - 1 - 2*r)
    if iterations < 1:
        raise ValueError, "Set iterations=1 or more."
    for iteration in xrange(iterations):
        if (xc + r - x0 < 0.1 and yc + r - y0 < 0.1):
            break  # Accurate enough.
        # Start with whole-pixel shifts.
        if abs(xc - x0 - r) >= 0.6:
            x0 = np.clip(round(xc) - r, *xbounds)
            x1 = x0 + 2*r + 1
        if abs(yc - y0 -r) >= 0.6:
            y0 = np.clip(round(yc) - r, *ybounds)
            y1 = y0 + 2*r + 1
        # if abs(xc - x0 - r) < 0.6 and (yc -y0 -r) < 0.6:
            # Subpixel interpolation using a second-order spline.
            # interpolation.shift(neighborhood,[yc, xc],mode='constant',cval=0., order=2)
        neighborhood = circular_mask(diameter)*bp_image[y0:y1, x0:x1]    
        yc, xc = measurements.center_of_mass(neighborhood)  # neighborhood coordinates
        yc, xc = yc + y0, xc + x0  # image coords
    
    # Characterize the neighborhood of our final centroid.
    mass = neighborhood.sum() 
    Rg = np.sqrt(np.sum(rgmask(diameter)*neighborhood)/mass)
    ecc = np.sqrt((np.sum(neighborhood*cosmask(diameter)))**2 + 
                  (np.sum(neighborhood*sinmask(diameter)))**2) / \
                  (mass - neighborhood[r, r] + 1e-6)
    raw_neighborhood = raw_image[y0:y1, x0:x1][circular_mask(diameter)]
    signal = raw_neighborhood.max() # black_level subtracted later
    return Series([xc, yc, mass, Rg, ecc, signal])

def locate(image, diameter, minmass=100., separation=None, 
           noise_size=1, smoothing_size=None, threshold=1, invert=False,
           percentile=64, pickN=None, preprocess=True):
    """Read an image, do optional image preparation and cleanup, and locate 
    Gaussian-like blobs of a given size above a given total brightness.

    Parameters
    ----------
    image: image array
    diameter : feature size in px
    minmass : minimum integrated brightness
       Default is 100, but a good value is often much higher. This is a 
       crucial parameter for elminating spurrious features.
    separation : feature separation in px
    noise_size : scale of Gaussian blurring. Default 1.
    smoothing_size : defauls to separation
    threshold : Clip bandpass result below this value.
        Default 1; use 8 for 16-bit images.
    invert : Set to True if features are darker than background. False by
        default.
    percentile : Features must have a peak brighter than pixels in this
        percentile. This helps eliminate spurrious peaks.
    pickN : Not Implemented
    preprocess : Set to False to turn out automatic preprocessing.

    Returns
    -------
    DataFrame([x, y, mass, size, ecc, signal])

    where mass means total integrated brightness of the blob
    and size means the radius of gyration of its Gaussian-like profile
    """
    # Validate parameters.
    if not diameter & 1:
        raise ValueError, "Feature diameter must be an odd number. Round up."
    if not separation:
        separation = diameter + 1
    smoothing_size = smoothing_size if smoothing_size else diameter # default
    if preprocess:
        if invert:
            # Tempting to do this in place, but if it is called multiple
            # times on the same image, chaos reigns.
            max_value = np.iinfo(image.dtype).max
            image = image ^ max_value
        bp_image = bandpass(image, noise_size, smoothing_size, threshold)
    else:
        bp_image = image.copy()
    bp_image = scale_to_gamut(bp_image, image.dtype)

    f = DataFrame(local_maxima(bp_image, diameter, separation, percentile),
                  columns=['x', 'y'])
    approx_mass = f.apply(
        lambda x: estimate_mass(bp_image, x[0], x[1], diameter), axis=1)
    count_local_maxima = len(f)
    f = f[approx_mass > minmass].apply(
        lambda x: refine_centroid(image, bp_image, x[0], x[1], diameter, minmass), 
        axis=1)
    logger.info("%s local maxima, %s of qualifying mass", 
                count_local_maxima, len(f)) 
    columns = ['x', 'y', 'mass', 'size', 'ecc', 'signal']
    if len(f) == 0:
        return DataFrame(columns=columns) # empty
    f.columns = columns
    black_level, noise = uncertainty.measure_noise(image, diameter, threshold)
    f['signal'] -= black_level
    ep = uncertainty.static_error(f, noise, diameter, noise_size)
    f = f.join(ep)
    return f

def batch(frames, diameter, minmass=100, separation=None,
          noise_size=1, smoothing_size=None, threshold=1, invert=False,
          percentile=64, pickN=None, preprocess=True, 
          store=None, conn=None, sql_flavor=None, table=None,
          do_not_return=False):
    """Process a list of images, doing optional image preparation and cleanup, 
    locating Gaussian-like blobs of a given size above a given total brightness.

    Parameters
    ----------
    frames : iterable frames
        For example, frames = mr.video.frame_generator('video_file.avi')
                  or frames = [array1, array2, array3]
    diameter : feature size in px
    minmass : minimum integrated brightness
       Default is 100, but a good value is often much higher. This is a 
       crucial parameter for elminating spurrious features.
    separation : feature separation in px. Default = 1 + diamter.
    noise_size : scale of Gaussian blurring. Default = 1.
    smoothing_size : Default = separation.
    threshold : Clip bandpass result below this value.
        Default 1; use 8 for 16-bit images.
    invert : Set to True if features are darker than background. False by
        default.
    percentile : Features must have a peak brighter than pixels in this
        percentile. This helps eliminate spurrious peaks.
    pickN : Not Implemented
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
                   smoothing_size, invert, percentile, pickN, 
                   pd.Timestamp(timestamp)], 
                  index=['source', 
                         'diameter', 'minmass', 'separation', 'noise_size', 
                         'smoothing_size', 'invert', 'percentile', 'pickN',
                         'timestamp'])
    all_centroids = [] 
    for i, image in enumerate(frames):
        # If frames has a cursor property, use it. Otherwise, just count
        # the frames from 0.
        try:
            frame_no = frames.cursor - 1
        except AttributeError:
            frame_no = i 
        centroids = locate(image, diameter, minmass, separation, 
                           noise_size, smoothing_size, threshold, invert,
                           percentile, pickN, preprocess)
        centroids['frame'] = frame_no
        logger.info("Frame %d: %d features", frame_no, len(centroids))
        if len(centroids) == 0:
            continue
        indexed = ['frame'] # columns on which you can perform queries
        if store is not None:
            store.append(table, centroids, data_columns=indexed)
            store.flush() # Force save. Not essential.
        elif conn is not None:
            if sql_flavor is None:
                raise ValueError, \
                    "Specifiy sql_flavor: MySQL or sqlite."
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
