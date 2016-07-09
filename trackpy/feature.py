from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import warnings
import logging

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
from pandas import DataFrame

from .preprocessing import bandpass, scale_to_gamut, scalefactor_to_gamut
from .utils import record_meta, validate_tuple
from .masks import (binary_mask, N_binary_mask, r_squared_mask,
                    x_squared_masks, cosmask, sinmask)
from .uncertainty import _static_error, measure_noise
import trackpy  # to get trackpy.__version__

from .try_numba import NUMBA_AVAILABLE
from .feature_numba import (_numba_refine_2D, _numba_refine_2D_c,
                            _numba_refine_2D_c_a, _numba_refine_3D)

logger = logging.getLogger(__name__)


def percentile_threshold(image, percentile):
    """Find grayscale threshold based on distribution in image."""

    not_black = image[np.nonzero(image)]
    if len(not_black) == 0:
        return np.nan
    return np.percentile(not_black, percentile)


def minmass_version_change(raw_image, old_minmass, preprocess=True,
                           invert=False, noise_size=1, smoothing_size=None,
                           threshold=None):
    """Convert minmass value from v0.2.4 to v0.3.

    From trackpy version 0.3.0, the mass calculation is changed. Before
    version 0.3.0 the mass was calculated from a rescaled image. From version
    0.3.0, this rescaling is compensated at the end so that the mass reflects
    the actual intensities in the image.

    This function calculates the scalefactor between the old and new mass
    and applies it to calculate the new minmass filter value.

    Parameters
    ----------
    raw_image : ndarray
    old_minmass : number
    preprocess : boolean, optional
        Defaults to True
    invert : boolean, optional
        Defaults to False
    noise_size : number, optional
        Defaults to 1
    smoothing_size : number, optional
        Required when preprocessing. In locate, it equals diameter by default.
    threshold : number, optional

    Returns
    -------
    New minmass
    """
    if preprocess and smoothing_size is None:
        raise ValueError('Please specify the smoothing size. By default, this '
                         'equals diameter.')

    if np.issubdtype(raw_image.dtype, np.integer):
        dtype = raw_image.dtype
        if invert:
            raw_image = raw_image ^ np.iinfo(dtype).max
    else:
        dtype = np.uint8
        if invert:
            raw_image = 1 - raw_image

    if preprocess:
        image = bandpass(raw_image, noise_size, smoothing_size, threshold)
    else:
        image = raw_image

    scale_factor = scalefactor_to_gamut(image, dtype)

    return int(old_minmass / scale_factor)


def local_maxima(image, radius, percentile=64, margin=None):
    """Find local maxima whose brightness is above a given percentile.

    Parameters
    ----------
    radius : integer definition of "local" in "local maxima"
    percentile : chooses minimum grayscale value for a local maximum
    margin : zone of exclusion at edges of image. Defaults to radius.
            A smarter value is set by locate().
    """
    if margin is None:
        margin = radius

    ndim = image.ndim
    # Compute a threshold based on percentile.
    threshold = percentile_threshold(image, percentile)
    if np.isnan(threshold):
        warnings.warn("Image is completely black.", UserWarning)
        return np.empty((0, ndim))

    # The intersection of the image with its dilation gives local maxima.
    if not np.issubdtype(image.dtype, np.integer):
        raise TypeError("Perform dilation on exact (i.e., integer) data.")
    footprint = binary_mask(radius, ndim)
    dilation = ndimage.grey_dilation(image, footprint=footprint,
                                     mode='constant')
    maxima = np.vstack(np.where((image == dilation) & (image > threshold))).T
    if not np.size(maxima) > 0:
        warnings.warn("Image contains no local maxima.", UserWarning)
        return np.empty((0, ndim))

    # Do not accept peaks near the edges.
    shape = np.array(image.shape)
    near_edge = np.any((maxima < margin) | (maxima > (shape - margin - 1)), 1)
    maxima = maxima[~near_edge]
    if not np.size(maxima) > 0:
        warnings.warn("All local maxima were in the margins.", UserWarning)

    # Return coords in as a numpy array shaped so it can be passed directly
    # to the DataFrame constructor.
    return maxima


def estimate_mass(image, radius, coord):
    "Compute the total brightness in the neighborhood of a local maximum."
    square = [slice(c - rad, c + rad + 1) for c, rad in zip(coord, radius)]
    neighborhood = binary_mask(radius, image.ndim)*image[square]
    return np.sum(neighborhood)


def estimate_size(image, radius, coord, estimated_mass):
    "Compute the total brightness in the neighborhood of a local maximum."
    square = [slice(c - rad, c + rad + 1) for c, rad in zip(coord, radius)]
    neighborhood = binary_mask(radius, image.ndim)*image[square]
    Rg = np.sqrt(np.sum(r_squared_mask(radius, image.ndim) * neighborhood) /
                 estimated_mass)
    return Rg


def _safe_center_of_mass(x, radius, grids):
    normalizer = x.sum()
    if normalizer == 0:  # avoid divide-by-zero errors
        return np.array(radius)
    return np.array([(x * grids[dim]).sum() / normalizer
                    for dim in range(x.ndim)])


def refine(raw_image, image, radius, coords, separation=0, max_iterations=10,
           engine='auto', shift_thresh=0.6, break_thresh=None,
           characterize=True, walkthrough=False):
    """Find the center of mass of a bright feature starting from an estimate.

    Characterize the neighborhood of a local maximum, and iteratively
    hone in on its center-of-brightness. Return its coordinates, integrated
    brightness, size (Rg), eccentricity (0=circular), and signal strength.

    Parameters
    ----------
    raw_image : array (any dimensions)
        used for final characterization
    image : array (any dimension)
        processed image, used for locating center of mass
    coord : array
        estimated position
    separation : float or tuple
        Minimum separtion between features.
        Default is 0. May be a tuple, see diameter for details.
    max_iterations : integer
        max number of loops to refine the center of mass, default 10
    engine : {'python', 'numba'}
        Numba is faster if available, but it cannot do walkthrough.
    shift_thresh : float, optional
        Default 0.6 (unit is pixels).
        If the brightness centroid is more than this far off the mask center,
        shift mask to neighboring pixel. The new mask will be used for any
        remaining iterations.
    break_thresh : float, optional
        Deprecated
    characterize : boolean, True by default
        Compute and return mass, size, eccentricity, signal.
    walkthrough : boolean, False by default
        Print the offset on each loop and display final neighborhood image.
    """
    if break_thresh is not None:
        warnings.warn("break_threshold will be deprecated: shift_threshold is"
                      "the only parameter that determines when to shift the"
                      "mask.")
    if max_iterations <= 0:
        warnings.warn("max_iterations has to be larger than 0. setting it to 1.")
        max_iterations = 1
    # ensure that radius is tuple of integers, for direct calls to refine()
    radius = validate_tuple(radius, image.ndim)
    separation = validate_tuple(separation, image.ndim)
    # Main loop will be performed in separate function.
    if engine == 'auto':
        if NUMBA_AVAILABLE and image.ndim in [2, 3]:
            engine = 'numba'
        else:
            engine = 'python'

    # In here, coord is an integer. Make a copy, will not modify inplace.
    coords = np.round(coords).astype(np.int)

    if engine == 'python':
        results = _refine(raw_image, image, radius, coords, max_iterations,
                          shift_thresh, characterize, walkthrough)
    elif engine == 'numba':
        if not NUMBA_AVAILABLE:
            warnings.warn("numba could not be imported. Without it, the "
                          "'numba' engine runs very slow. Use the 'python' "
                          "engine or install numba.", UserWarning)
        if image.ndim not in [2, 3]:
            raise NotImplementedError("The numba engine only supports 2D or 3D "
                                      "images. You can extend it if you feel "
                                      "like a hero.")
        if walkthrough:
            raise ValueError("walkthrough is not availabe in the numba engine")
        # Do some extra prep in pure Python that can't be done in numba.
        N = coords.shape[0]
        mask = binary_mask(radius, image.ndim)
        if image.ndim == 3:
            if characterize:
                if np.all(radius[1:] == radius[:-1]):
                    results_columns = 8
                else:
                    results_columns = 10
            else:
                results_columns = 4
            r2_mask = r_squared_mask(radius, image.ndim)[mask]
            x2_masks = x_squared_masks(radius, image.ndim)
            z2_mask = image.ndim * x2_masks[0][mask]
            y2_mask = image.ndim * x2_masks[1][mask]
            x2_mask = image.ndim * x2_masks[2][mask]
            results = np.empty((N, results_columns), dtype=np.float64)
            maskZ, maskY, maskX = np.asarray(np.asarray(mask.nonzero()),
                                             dtype=np.int16)
            _numba_refine_3D(np.asarray(raw_image), np.asarray(image),
                             radius[0], radius[1], radius[2], coords, N,
                             int(max_iterations), shift_thresh,
                             characterize,
                             image.shape[0], image.shape[1], image.shape[2],
                             maskZ, maskY, maskX, maskX.shape[0],
                             r2_mask, z2_mask, y2_mask, x2_mask, results)
        elif not characterize:
            mask_coordsY, mask_coordsX = np.asarray(mask.nonzero(), np.int16)
            results = np.empty((N, 3), dtype=np.float64)
            _numba_refine_2D(np.asarray(image), radius[0], radius[1], coords, N,
                             int(max_iterations), shift_thresh,
                             image.shape[0], image.shape[1],
                             mask_coordsY, mask_coordsX, mask_coordsY.shape[0],
                             results)
        elif radius[0] == radius[1]:
            mask_coordsY, mask_coordsX = np.asarray(mask.nonzero(), np.int16)
            results = np.empty((N, 7), dtype=np.float64)
            r2_mask = r_squared_mask(radius, image.ndim)[mask]
            cmask = cosmask(radius)[mask]
            smask = sinmask(radius)[mask]
            _numba_refine_2D_c(np.asarray(raw_image), np.asarray(image),
                               radius[0], radius[1], coords, N,
                               int(max_iterations), shift_thresh,
                               image.shape[0], image.shape[1], mask_coordsY,
                               mask_coordsX, mask_coordsY.shape[0],
                               r2_mask, cmask, smask, results)
        else:
            mask_coordsY, mask_coordsX = np.asarray(mask.nonzero(), np.int16)
            results = np.empty((N, 8), dtype=np.float64)
            x2_masks = x_squared_masks(radius, image.ndim)
            y2_mask = image.ndim * x2_masks[0][mask]
            x2_mask = image.ndim * x2_masks[1][mask]
            cmask = cosmask(radius)[mask]
            smask = sinmask(radius)[mask]
            _numba_refine_2D_c_a(np.asarray(raw_image), np.asarray(image),
                                 radius[0], radius[1], coords, N,
                                 int(max_iterations), shift_thresh,
                                 image.shape[0], image.shape[1], mask_coordsY,
                                 mask_coordsX, mask_coordsY.shape[0],
                                 y2_mask, x2_mask, cmask, smask, results)
    else:
        raise ValueError("Available engines are 'python' and 'numba'")

    # Flat peaks return multiple nearby maxima. Eliminate duplicates.
    if np.all(np.greater(separation, 0)):
        mass_index = image.ndim  # i.e., index of the 'mass' column
        while True:
            # Rescale positions, so that pairs are identified below a distance
            # of 1. Do so every iteration (room for improvement?)
            positions = results[:, :mass_index]/list(reversed(separation))
            mass = results[:, mass_index]
            duplicates = cKDTree(positions, 30).query_pairs(1)
            if len(duplicates) == 0:
                break
            to_drop = []
            for p0, p1 in duplicates:
                # Drop the dimmer one.
                m0, m1 = mass[p0], mass[p1]
                if m0 < m1:
                    to_drop.append(p0)
                elif m0 > m1:
                    to_drop.append(p1)
                else:
                    # Rare corner case: a tie!
                    # Break ties by sorting by sum of coordinates, to avoid
                    # any randomness resulting from cKDTree returning a set.
                    to_drop.append([p0, p1][np.argmin(np.sum(positions.take([p0, p1], 0), 1))])
            results = np.delete(results, to_drop, 0)

    return results


# (This is pure Python. A numba variant follows below.)
def _refine(raw_image, image, radius, coords, max_iterations,
            shift_thresh, characterize, walkthrough):
    if not np.issubdtype(coords.dtype, np.int):
        raise ValueError('The coords array should be of integer datatype')
    ndim = image.ndim
    isotropic = np.all(radius[1:] == radius[:-1])
    mask = binary_mask(radius, ndim).astype(np.uint8)

    # Declare arrays that we will fill iteratively through loop.
    N = coords.shape[0]
    final_coords = np.empty_like(coords, dtype=np.float64)
    mass = np.empty(N, dtype=np.float64)
    raw_mass = np.empty(N, dtype=np.float64)
    if characterize:
        if isotropic:
            Rg = np.empty(N, dtype=np.float64)
        else:
            Rg = np.empty((N, len(radius)), dtype=np.float64)
        ecc = np.empty(N, dtype=np.float64)
        signal = np.empty(N, dtype=np.float64)

    ogrid = np.ogrid[[slice(0, i) for i in mask.shape]]  # for center of mass
    ogrid = [g.astype(float) for g in ogrid]

    for feat, coord in enumerate(coords):
        for iteration in range(max_iterations):
            # Define the circular neighborhood of (x, y).
            rect = [slice(c - r, c + r + 1) for c, r in zip(coord, radius)]
            neighborhood = mask*image[rect]
            cm_n = _safe_center_of_mass(neighborhood, radius, ogrid)
            cm_i = cm_n - radius + coord  # image coords

            off_center = cm_n - radius
            logger.debug('off_center: %f', off_center)
            if np.all(np.abs(off_center) < shift_thresh):
                break  # Accurate enough.
            # If we're off by more than half a pixel in any direction, move..
            coord[off_center > shift_thresh] += 1
            coord[off_center < -shift_thresh] -= 1
            # Don't move outside the image!
            upper_bound = np.array(image.shape) - 1 - radius
            coord = np.clip(coord, radius, upper_bound).astype(int)

        # matplotlib and ndimage have opposite conventions for xy <-> yx.
        final_coords[feat] = cm_i[..., ::-1]

        if walkthrough:
            import matplotlib.pyplot as plt
            plt.imshow(neighborhood)

        # Characterize the neighborhood of our final centroid.
        mass[feat] = neighborhood.sum()
        if not characterize:
            continue  # short-circuit loop
        if isotropic:
            Rg[feat] = np.sqrt(np.sum(r_squared_mask(radius, ndim) *
                                      neighborhood) / mass[feat])
        else:
            Rg[feat] = np.sqrt(ndim * np.sum(x_squared_masks(radius, ndim) *
                                             neighborhood,
                                             axis=tuple(range(1, ndim + 1))) /
                               mass[feat])[::-1]  # change order yx -> xy
        # I only know how to measure eccentricity in 2D.
        if ndim == 2:
            ecc[feat] = np.sqrt(np.sum(neighborhood*cosmask(radius))**2 +
                                np.sum(neighborhood*sinmask(radius))**2)
            ecc[feat] /= (mass[feat] - neighborhood[radius] + 1e-6)
        else:
            ecc[feat] = np.nan
        signal[feat] = neighborhood.max()  # based on bandpassed image
        raw_neighborhood = mask*raw_image[rect]
        raw_mass[feat] = raw_neighborhood.sum()  # based on raw image

    if not characterize:
        return np.column_stack([final_coords, mass])
    else:
        return np.column_stack([final_coords, mass, Rg, ecc, signal, raw_mass])


def locate(raw_image, diameter, minmass=None, maxsize=None, separation=None,
           noise_size=1, smoothing_size=None, threshold=None, invert=False,
           percentile=64, topn=None, preprocess=True, max_iterations=10,
           filter_before=True, filter_after=True,
           characterize=True, engine='auto'):
    """Locate Gaussian-like blobs of some approximate size in an image.

    Preprocess the image by performing a band pass and a threshold.
    Locate all peaks of brightness, characterize the neighborhoods of the peaks
    and take only those with given total brightness ("mass"). Finally,
    refine the positions of each peak.

    Parameters
    ----------
    image : array
         any N-dimensional image
    diameter : odd integer or tuple of odd integers
        This may be a single number or a tuple giving the feature's
        extent in each dimension, useful when the dimensions do not have
        equal resolution (e.g. confocal microscopy). The tuple order is the
        same as the image shape, conventionally (z, y, x) or (y, x). The
        number(s) must be odd integers. When in doubt, round up.
    minmass : float
        The minimum integrated brightness.
        Default is 100 for integer images and 1 for float images, but a good
        value is often much higher. This is a crucial parameter for eliminating
        spurious features.
        .. warning:: The mass value is changed since v0.3.0
    maxsize : float
        maximum radius-of-gyration of brightness, default None
    separation : float or tuple
        Minimum separtion between features.
        Default is diameter + 1. May be a tuple, see diameter for details.
    noise_size : float or tuple
        Width of Gaussian blurring kernel, in pixels
        Default is 1. May be a tuple, see diameter for details.
    smoothing_size : float or tuple
        Size of boxcar smoothing, in pixels
        Default is diameter. May be a tuple, see diameter for details.
    threshold : float
        Clip bandpass result below this value.
        Default, None, defers to default settings of the bandpass function.
    invert : boolean
        Set to True if features are darker than background. False by default.
    percentile : float
        Features must have a peak brighter than pixels in this
        percentile. This helps eliminate spurious peaks.
    topn : integer
        Return only the N brightest features above minmass.
        If None (default), return all features above minmass.
    preprocess : boolean
        Set to False to turn off bandpass preprocessing.
    max_iterations : integer
        max number of loops to refine the center of mass, default 10
    filter_before : boolean
        Use minmass (and maxsize, if set) to eliminate spurious features
        based on their estimated mass and size before refining position.
        Default (None) defers to trackpy, to optimize for performance.
    filter_after : boolean
        Use final characterizations of mass and size to eliminate spurious
        features. True by default.
    characterize : boolean
        Compute "extras": eccentricity, signal, ep. True by default.
    engine : {'auto', 'python', 'numba'}

    Returns
    -------
    DataFrame([x, y, mass, size, ecc, signal])
        where mass means total integrated brightness of the blob,
        size means the radius of gyration of its Gaussian-like profile,
        and ecc is its eccentricity (0 is circular).

    See Also
    --------
    batch : performs location on many images in batch
    minmass_version_change : to convert minmass from v0.2.4 to v0.3.0

    Notes
    -----
    Locate works with a coordinate system that has its origin at the center of
    pixel (0, 0). In almost all cases this will be the topleft pixel: the
    y-axis is pointing downwards.

    This is an implementation of the Crocker-Grier centroid-finding algorithm.
    [1]_

    References
    ----------
    .. [1] Crocker, J.C., Grier, D.G. http://dx.doi.org/10.1006/jcis.1996.0217

    """

    # Validate parameters and set defaults.
    raw_image = np.squeeze(raw_image)
    shape = raw_image.shape
    ndim = len(shape)
    if filter_before is None:
        # TODO smarter perf optimization, see GH issue #141
        filter_before = False

    diameter = validate_tuple(diameter, ndim)
    diameter = tuple([int(x) for x in diameter])
    if not np.all([x & 1 for x in diameter]):
        raise ValueError("Feature diameter must be an odd integer. Round up.")
    radius = tuple([x//2 for x in diameter])

    isotropic = np.all(radius[1:] == radius[:-1])
    if (not isotropic) and (maxsize is not None):
        raise ValueError("Filtering by size is not available for anisotropic "
                         "features.")

    if separation is None:
        separation = tuple([x + 1 for x in diameter])
    else:
        separation = validate_tuple(separation, ndim)

    if smoothing_size is None:
        smoothing_size = diameter
    else:
        smoothing_size = validate_tuple(smoothing_size, ndim)

    noise_size = validate_tuple(noise_size, ndim)

    # Check whether the image looks suspiciously like a color image.
    if 3 in shape or 4 in shape:
        dim = raw_image.ndim
        warnings.warn("I am interpreting the image as {0}-dimensional. "
                      "If it is actually a {1}-dimensional color image, "
                      "convert it to grayscale first.".format(dim, dim-1))

    if minmass is None:
        if np.issubdtype(raw_image.dtype, np.integer):
            minmass = 100
        else:
            minmass = 1.

    # Determine `image`: the image to find the local maxima on
    if preprocess:
        if invert:
            # It is tempting to do this in place, but if it is called multiple
            # times on the same image, chaos reigns.
            if np.issubdtype(raw_image.dtype, np.integer):
                max_value = np.iinfo(raw_image.dtype).max
                raw_image = raw_image ^ max_value
            else:
                # To avoid degrading performance, assume gamut is zero to one.
                # Have you ever encountered an image of unnormalized floats?
                raw_image = 1 - raw_image
        image = bandpass(raw_image, noise_size, smoothing_size, threshold)

        # Coerce the image into integer type. Rescale to fill dynamic range.
        if np.issubdtype(raw_image.dtype, np.integer):
            dtype = raw_image.dtype
        else:
            dtype = np.uint8
        scale_factor = scalefactor_to_gamut(image, dtype)
        image = scale_to_gamut(image, dtype, scale_factor)
    elif np.issubdtype(raw_image.dtype, np.integer):
        # Do nothing when image is already of integer type
        scale_factor = 1.
        image = raw_image
    else:
        # Coerce the image into uint8 type. Rescale to fill dynamic range.
        scale_factor = scalefactor_to_gamut(raw_image, np.uint8)
        image = scale_to_gamut(raw_image, np.uint8, scale_factor)

    # Set up a DataFrame for the final results.
    if image.ndim < 4:
        coord_columns = ['x', 'y', 'z'][:image.ndim]
    else:
        coord_columns = map(lambda i: 'x' + str(i), range(image.ndim))
    MASS_COLUMN_INDEX = len(coord_columns)
    columns = coord_columns + ['mass']
    if characterize:
        if isotropic:
            SIZE_COLUMN_INDEX = len(columns)
            columns += ['size']
        else:
            SIZE_COLUMN_INDEX = range(len(columns),
                                      len(columns) + len(coord_columns))
            columns += ['size_' + cc for cc in coord_columns]
        SIGNAL_COLUMN_INDEX = len(columns) + 1
        columns += ['ecc', 'signal', 'raw_mass']
        if isotropic and np.all(noise_size[1:] == noise_size[:-1]):
            columns += ['ep']
        else:
            columns += ['ep_' + cc for cc in coord_columns]

    # Find local maxima.
    # Define zone of exclusion at edges of image, avoiding
    #   - Features with incomplete image data ("radius")
    #   - Extended particles that cannot be explored during subpixel
    #       refinement ("separation")
    #   - Invalid output of the bandpass step ("smoothing_size")
    margin = tuple([max(rad, sep // 2 - 1, sm // 2) for (rad, sep, sm) in
                    zip(radius, separation, smoothing_size)])
    coords = local_maxima(image, radius, percentile, margin)
    count_maxima = coords.shape[0]

    if count_maxima == 0:
        return DataFrame(columns=columns)

    # Proactively filter based on estimated mass/size before
    # refining positions.
    if filter_before:
        approx_mass = np.empty(count_maxima)  # initialize to avoid appending
        for i in range(count_maxima):
            approx_mass[i] = estimate_mass(image, radius, coords[i])
        condition = approx_mass > minmass * scale_factor
        if maxsize is not None:
            approx_size = np.empty(count_maxima)
            for i in range(count_maxima):
                approx_size[i] = estimate_size(image, radius, coords[i],
                                               approx_mass[i])
            condition &= approx_size < maxsize
        coords = coords[condition]
    count_qualified = coords.shape[0]

    if count_qualified == 0:
        warnings.warn("No maxima survived mass- and size-based prefiltering. "
                      "Be advised that the mass computation was changed from "
                      "version 0.2.4 to 0.3.0. See the documentation and the "
                      "convenience function minmass_version_change.")
        return DataFrame(columns=columns)

    # Refine their locations and characterize mass, size, etc.
    refined_coords = refine(raw_image, image, radius, coords,
                            separation=separation, max_iterations=max_iterations,
                            engine=engine, characterize=characterize)
    # mass and signal values has to be corrected due to the rescaling
    # raw_mass was obtained from raw image; size and ecc are scale-independent
    refined_coords[:, MASS_COLUMN_INDEX] *= 1. / scale_factor
    if characterize:
        refined_coords[:, SIGNAL_COLUMN_INDEX] *= 1. / scale_factor

    # Filter again, using final ("exact") mass -- and size, if set.
    exact_mass = refined_coords[:, MASS_COLUMN_INDEX]
    if filter_after:
        condition = exact_mass > minmass
        if maxsize is not None:
            exact_size = refined_coords[:, SIZE_COLUMN_INDEX]
            condition &= exact_size < maxsize
        refined_coords = refined_coords[condition]
        exact_mass = exact_mass[condition]  # used below by topn
    count_qualified = refined_coords.shape[0]

    if count_qualified == 0:
        warnings.warn("No maxima survived mass- and size-based filtering. "
                      "Be advised that the mass computation was changed from "
                      "version 0.2.4 to 0.3.0. See the documentation and the "
                      "convenience function minmass_version_change.")
        return DataFrame(columns=columns)

    if topn is not None and count_qualified > topn:
        if topn == 1:
            # special case for high performance and correct shape
            refined_coords = refined_coords[np.argmax(exact_mass)]
            refined_coords = refined_coords.reshape(1, -1)
        else:
            refined_coords = refined_coords[np.argsort(exact_mass)][-topn:]

    # Estimate the uncertainty in position using signal (measured in refine)
    # and noise (measured here below).
    if characterize:
        if preprocess:  # identify background regions from the processed image
            black_level, noise = measure_noise(image, raw_image, radius)
        else:  # identify background regions from the provided image
            black_level, noise = measure_noise(image, raw_image, radius)
        Npx = N_binary_mask(radius, ndim)
        mass = refined_coords[:, SIGNAL_COLUMN_INDEX + 1] - Npx * black_level
        ep = _static_error(mass, noise, radius[::-1], noise_size[::-1])
        refined_coords = np.column_stack([refined_coords, ep])

    f = DataFrame(refined_coords, columns=columns)

    # If this is a pims Frame object, it has a frame number.
    # Tag it on; this is helpful for parallelization.
    if hasattr(raw_image, 'frame_no') and raw_image.frame_no is not None:
        f['frame'] = raw_image.frame_no
    return f


def batch(frames, diameter, minmass=100, maxsize=None, separation=None,
          noise_size=1, smoothing_size=None, threshold=None, invert=False,
          percentile=64, topn=None, preprocess=True, max_iterations=10,
          filter_before=None, filter_after=True,
          characterize=True, engine='auto',
          output=None, meta=None):
    """Locate Gaussian-like blobs of some approximate size in a set of images.

    Preprocess the image by performing a band pass and a threshold.
    Locate all peaks of brightness, characterize the neighborhoods of the peaks
    and take only those with given total brightness ("mass"). Finally,
    refine the positions of each peak.

    Parameters
    ----------
    frames : list (or iterable) of images
    diameter : odd integer or tuple of odd integers
        This may be a single number or a tuple giving the feature's
        extent in each dimension, useful when the dimensions do not have
        equal resolution (e.g. confocal microscopy). The tuple order is the
        same as the image shape, conventionally (z, y, x) or (y, x). The
        number(s) must be odd integers. When in doubt, round up.
    minmass : float
        The minimum integrated brightness.
        Default is 100 for integer images and 1 for float images, but a good
        value is often much higher. This is a crucial parameter for eliminating
        spurious features.
        .. warning:: The mass value was changed since v0.3.0
    maxsize : float
        maximum radius-of-gyration of brightness, default None
    separation : float or tuple
        Minimum separtion between features.
        Default is diameter + 1. May be a tuple, see diameter for details.
    noise_size : float or tuple
        Width of Gaussian blurring kernel, in pixels
        Default is 1. May be a tuple, see diameter for details.
    smoothing_size : float or tuple
        Size of boxcar smoothing, in pixels
        Default is diameter. May be a tuple, see diameter for details.
    threshold : float
        Clip bandpass result below this value.
        Default, None, defers to default settings of the bandpass function.
    invert : boolean
        Set to True if features are darker than background. False by default.
    percentile : float
        Features must have a peak brighter than pixels in this
        percentile. This helps eliminate spurious peaks.
    topn : integer
        Return only the N brightest features above minmass.
        If None (default), return all features above minmass.
    preprocess : boolean
        Set to False to turn off bandpass preprocessing.
    max_iterations : integer
        max number of loops to refine the center of mass, default 10
    filter_before : boolean
        Use minmass (and maxsize, if set) to eliminate spurious features
        based on their estimated mass and size before refining position.
        Default (None) defers to trackpy, to optimize for performance.
    filter_after : boolean
        Use final characterizations of mass and size to eliminate spurious
        features. True by default.
    characterize : boolean
        Compute "extras": eccentricity, signal, ep. True by default.
    engine : {'auto', 'python', 'numba'}
    output : {None, trackpy.PandasHDFStore, SomeCustomClass}
        If None, return all results as one big DataFrame. Otherwise, pass
        results from each frame, one at a time, to the put() method
        of whatever class is specified here.
    meta : filepath or file object, optional
        If specified, information relevant to reproducing this batch is saved
        as a YAML file, a plain-text machine- and human-readable format.
        By default, this is None, and no file is saved.

    Returns
    -------
    DataFrame([x, y, mass, size, ecc, signal])
        where mass means total integrated brightness of the blob,
        size means the radius of gyration of its Gaussian-like profile,
        and ecc is its eccentricity (0 is circular).

    See Also
    --------
    locate : performs location on a single image
    minmass_version_change : to convert minmass from v0.2.4 to v0.3.0

    Notes
    -----
    This is an implementation of the Crocker-Grier centroid-finding algorithm.
    [1]_

    Locate works with a coordinate system that has its origin at the center of
    pixel (0, 0). In almost all cases this will be the topleft pixel: the
    y-axis is pointing downwards.

    References
    ----------
    .. [1] Crocker, J.C., Grier, D.G. http://dx.doi.org/10.1006/jcis.1996.0217

    """
    # Gather meta information and save as YAML in current directory.
    timestamp = pd.datetime.utcnow().strftime('%Y-%m-%d-%H%M%S')
    try:
        source = frames.filename
    except:
        source = None
    meta_info = dict(timestamp=timestamp,
                     trackpy_version=trackpy.__version__,
                     source=source, diameter=diameter, minmass=minmass,
                     maxsize=maxsize, separation=separation,
                     noise_size=noise_size, smoothing_size=smoothing_size,
                     invert=invert, percentile=percentile, topn=topn,
                     preprocess=preprocess, max_iterations=max_iterations,
                     filter_before=filter_before, filter_after=filter_after)

    if meta:
        if isinstance(meta, six.string_types):
            with open(meta, 'w') as file_obj:
                record_meta(meta_info, file_obj)
        else:
            # Interpret meta to be a file handle.
            record_meta(meta_info, meta)

    all_features = []
    for i, image in enumerate(frames):
        features = locate(image, diameter, minmass, maxsize, separation,
                          noise_size, smoothing_size, threshold, invert,
                          percentile, topn, preprocess, max_iterations,
                          filter_before, filter_after, characterize,
                          engine)
        if hasattr(image, 'frame_no') and image.frame_no is not None:
            frame_no = image.frame_no
            # If this works, locate created a 'frame' column.
        else:
            frame_no = i
            features['frame'] = i  # just counting iterations
        logger.info("Frame %d: %d features", frame_no, len(features))
        if len(features) == 0:
            continue

        if output is None:
            all_features.append(features)
        else:
            output.put(features)

    if output is None:
        if len(all_features) > 0:
            return pd.concat(all_features).reset_index(drop=True)
        else:  # return empty DataFrame
            warnings.warn("No maxima found in any frame.")
            return pd.DataFrame(columns=list(features.columns) + ['frame'])
    else:
        return output
