import warnings
import logging
from functools import partial

import numpy as np
import pandas as pd

from .preprocessing import (bandpass, convert_to_int, invert_image,
                            scalefactor_to_gamut)
from .utils import (record_meta, validate_tuple, is_isotropic,
                    default_pos_columns, default_size_columns,
                    pandas_concat, get_pool)
from .find import grey_dilation, where_close
from .refine import refine_com, refine_com_arr
from .masks import (binary_mask, N_binary_mask, r_squared_mask,
                    x_squared_masks, cosmask, sinmask)
from .uncertainty import _static_error, measure_noise
import trackpy  # to get trackpy.__version__

logger = logging.getLogger(__name__)


def minmass_v03_change(raw_image, old_minmass, preprocess=True,
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

    return old_minmass / scale_factor


def minmass_v04_change(raw_image, old_minmass, diameter, preprocess=True,
                       old_smoothing_size=None, new_smoothing_size=None):
    """Convert minmass value from v0.3 to v0.4.

    From trackpy version 0.4.0, the default image preprocessing is changed.
    Before version 0.4.0 a blurred out image (rolling or boxcar average) with
    a circular kernel with radius `diameter` was subtracted from the image
    before refinement and mass calculation. From version 0.4.0, this has
    changed to a square kernel with sides `diameter`, more or less twice as
    small. This increases tracking accuracy, and reduces the mass.

    This function estimates this difference and applies it to calculate the
    new minmass value.

    Here the following approximate relation between the "real" mass of the
    feature and the mass apparent from the bandpassed image is used:

    .. math::

        M_{bp} = M_{real} \\left( 1 - \\frac{n_r}{n_{bp}} \\right)

    Where :math:`n_r` denotes the number of pixels in the (circular)
    refine mask and :math:`n_{bp}` the number of pixels in the (square)
    rolling average kernel.

    This follows from a simple model where the bandpassed image :math:`I_{bp}`
    relates to the "real" feature :math:`F` and the noise :math:`N` by:

    .. math::

        I_{bp} = F + N - \\left(N + \\frac{M_{real}}{n_{bp}} \\right)

    Parameters
    ----------
    raw_image : ndarray
    old_minmass : number
    diameter : number or tuple
        Odd-valued number that is used in locate
    preprocess : boolean, optional
        Defaults to True. Minmass is not changed when preprocess=False.
    old_smoothing_size : number or tuple, optional
        The smoothing size used in the old (pre v0.4) trackpy version (the
        radius of the circular kernel). Defaults to diameter.
    new_smoothing_size : number or tuple, optional
        The smoothing size used in the new (post v0.4) trackpy version (the
        size of the sides of the square kernel). Defaults to diameter.

    Returns
    -------
    New minmass
    """
    if not preprocess:
        return old_minmass

    ndim = raw_image.ndim
    diameter = validate_tuple(diameter, ndim)
    if old_smoothing_size is None:
        old_smoothing_size = diameter
    else:
        old_smoothing_size = validate_tuple(old_smoothing_size, ndim)
    if new_smoothing_size is None:
        new_smoothing_size = diameter
    else:
        new_smoothing_size = validate_tuple(new_smoothing_size, ndim)

    radius = tuple(int(d / 2) for d in diameter)
    old_bp_size = tuple(s * 2 + 1 for s in old_smoothing_size)

    n_px_refine = N_binary_mask(radius, ndim)
    n_px_old_bp = N_binary_mask(old_bp_size, ndim)
    n_px_new_bp = np.prod(new_smoothing_size)

    real_minmass = old_minmass / (1 - n_px_refine / n_px_old_bp)
    new_minmass = real_minmass * (1 - n_px_refine / n_px_new_bp)
    return new_minmass


def local_maxima(image, radius, percentile=64, margin=None):
    """Find local maxima whose brightness is above a given percentile.
    This function will be deprecated. Please use the routines in trackpy.find,
    with the minimum separation between features as second argument.

    Parameters
    ----------
    image : ndarray
        For best performance, provide an integer-type array. If the type is not
        of integer-type, the image will be normalized and coerced to uint8.
    radius : the radius of the circular grey dilation kernel, which is the
        minimum separation between maxima
    percentile : chooses minimum grayscale value for a local maximum
    margin : zone of exclusion at edges of image. Defaults to radius.
            A smarter value is set by locate().
    """
    warnings.warn("Local_maxima will be deprecated: please use routines in "
                  "trackpy.find", PendingDeprecationWarning)
    return grey_dilation(image, radius, percentile, margin)


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


def refine(*args, **kwargs):
    """
    Deprecated.

    See also
    --------
    trackpy.refine.refine_com
    """
    warnings.warn("trackpy.feature.refine is deprecated: please use routines in "
                  "trackpy.refine", PendingDeprecationWarning)
    return refine_com(*args, **kwargs)


def locate(raw_image, diameter, minmass=None, maxsize=None, separation=None,
           noise_size=1, smoothing_size=None, threshold=None, invert=False,
           percentile=64, topn=None, preprocess=True, max_iterations=10,
           filter_before=None, filter_after=None,
           characterize=True, engine='auto'):
    """Locate Gaussian-like blobs of some approximate size in an image.

    Preprocess the image by performing a band pass and a threshold.
    Locate all peaks of brightness, characterize the neighborhoods of the peaks
    and take only those with given total brightness ("mass"). Finally,
    refine the positions of each peak.

    Parameters
    ----------
    raw_image : array (any dimensions)
        Image used for final characterization. Ideally, pixel values of
        this image are not rescaled, but it can also be identical to
        ``image``.
    image : array (same size as raw_image)
        Processed image used for centroid-finding and most particle
        measurements.
    diameter : odd integer or tuple of odd integers
        This may be a single number or a tuple giving the feature's
        extent in each dimension, useful when the dimensions do not have
        equal resolution (e.g. confocal microscopy). The tuple order is the
        same as the image shape, conventionally (z, y, x) or (y, x). The
        number(s) must be odd integers. When in doubt, round up.
    minmass : float
        The minimum integrated brightness. This is a crucial parameter for
        eliminating spurious features.
        Recommended minimum values are 100 for integer images and 1 for float
        images. Defaults to 0 (no filtering).
        .. warning:: The mass value is changed since v0.3.0
        .. warning:: The default behaviour of minmass has changed since v0.4.0
    maxsize : float
        maximum radius-of-gyration of brightness, default None
    separation : float or tuple
        Minimum separation between features.
        Default is diameter + 1. May be a tuple, see diameter for details.
    noise_size : float or tuple
        Width of Gaussian blurring kernel, in pixels
        Default is 1. May be a tuple, see diameter for details.
    smoothing_size : float or tuple
        The size of the sides of the square kernel used in boxcar (rolling
        average) smoothing, in pixels
        Default is diameter. May be a tuple, making the kernel rectangular.
    threshold : float
        Clip bandpass result below this value. Thresholding is done on the
        already background-subtracted image.
        By default, 1 for integer images and 1/255 for float images.
    invert : boolean
        This will be deprecated. Use an appropriate PIMS pipeline to invert a
        Frame or FramesSequence.
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
        filter_before is no longer supported as it does not improve performance.
    filter_after : boolean
        This parameter has been deprecated: use minmass and maxsize.
    characterize : boolean
        Compute "extras": eccentricity, signal, ep. True by default.
    engine : {'auto', 'python', 'numba'}

    Returns
    -------
    DataFrame([x, y, mass, size, ecc, signal, raw_mass])
        where "x, y" are appropriate to the dimensionality of the image,
        mass means total integrated brightness of the blob,
        size means the radius of gyration of its Gaussian-like profile,
        ecc is its eccentricity (0 is circular),
        and raw_mass is the total integrated brightness in raw_image.

    See Also
    --------
    batch : performs location on many images in batch
    minmass_v03_change : to convert minmass from v0.2.4 to v0.3.0
    minmass_v04_change : to convert minmass from v0.3.x to v0.4.x

    Notes
    -----
    Locate works with a coordinate system that has its origin at the center of
    pixel (0, 0). In almost all cases this will be the top-left pixel: the
    y-axis is pointing downwards.

    This is an implementation of the Crocker-Grier centroid-finding algorithm.
    [1]_

    References
    ----------
    .. [1] Crocker, J.C., Grier, D.G. http://dx.doi.org/10.1006/jcis.1996.0217

    """
    if invert:
        warnings.warn("The invert argument will be deprecated. Use a PIMS "
                      "pipeline for this.", PendingDeprecationWarning)
    if filter_before is not None:
        raise ValueError("The filter_before argument is no longer supported as "
                         "it does not improve performance. Features are "
                         "filtered after refine.") # see GH issue #141
    if filter_after is not None:
        warnings.warn("The filter_after argument has been deprecated: it is "
                      "always on, unless minmass = None and maxsize = None.",
                      DeprecationWarning)

    # Validate parameters and set defaults.
    raw_image = np.squeeze(raw_image)
    shape = raw_image.shape
    ndim = len(shape)

    diameter = validate_tuple(diameter, ndim)
    diameter = tuple([int(x) for x in diameter])
    if not np.all([x & 1 for x in diameter]):
        raise ValueError("Feature diameter must be an odd integer. Round up.")
    radius = tuple([x//2 for x in diameter])

    isotropic = np.all(radius[1:] == radius[:-1])
    if (not isotropic) and (maxsize is not None):
        raise ValueError("Filtering by size is not available for anisotropic "
                         "features.")

    is_float_image = not np.issubdtype(raw_image.dtype, np.integer)

    if separation is None:
        separation = tuple([x + 1 for x in diameter])
    else:
        separation = validate_tuple(separation, ndim)

    if smoothing_size is None:
        smoothing_size = diameter
    else:
        smoothing_size = validate_tuple(smoothing_size, ndim)

    noise_size = validate_tuple(noise_size, ndim)

    if minmass is None:
        minmass = 0

    # Check whether the image looks suspiciously like a color image.
    if 3 in shape or 4 in shape:
        dim = raw_image.ndim
        warnings.warn("I am interpreting the image as {}-dimensional. "
                      "If it is actually a {}-dimensional color image, "
                      "convert it to grayscale first.".format(dim, dim-1))

    if threshold is None:
        if is_float_image:
            threshold = 1/255.
        else:
            threshold = 1

    # Invert the image if necessary
    if invert:
        raw_image = invert_image(raw_image)

    # Determine `image`: the image to find the local maxima on.
    if preprocess:
        image = bandpass(raw_image, noise_size, smoothing_size, threshold)
    else:
        image = raw_image

    # For optimal performance, performance, coerce the image dtype to integer.
    if is_float_image:  # For float images, assume bitdepth of 8.
        dtype = np.uint8
    else:   # For integer images, take original dtype
        dtype = raw_image.dtype
    # Normalize_to_int does nothing if image is already of integer type.
    scale_factor, image = convert_to_int(image, dtype)

    pos_columns = default_pos_columns(image.ndim)

    # Find local maxima.
    # Define zone of exclusion at edges of image, avoiding
    #   - Features with incomplete image data ("radius")
    #   - Extended particles that cannot be explored during subpixel
    #       refinement ("separation")
    #   - Invalid output of the bandpass step ("smoothing_size")
    margin = tuple([max(rad, sep // 2 - 1, sm // 2) for (rad, sep, sm) in
                    zip(radius, separation, smoothing_size)])
    # Find features with minimum separation distance of `separation`. This
    # excludes detection of small features close to large, bright features
    # using the `maxsize` argument.
    coords = grey_dilation(image, separation, percentile, margin, precise=False)

    # Refine their locations and characterize mass, size, etc.
    refined_coords = refine_com(raw_image, image, radius, coords,
                                max_iterations=max_iterations,
                                engine=engine, characterize=characterize)
    if len(refined_coords) == 0:
        return refined_coords

    # Flat peaks return multiple nearby maxima. Eliminate duplicates.
    if np.all(np.greater(separation, 0)):
        to_drop = where_close(refined_coords[pos_columns], separation,
                              refined_coords['mass'])
        refined_coords.drop(to_drop, axis=0, inplace=True)
        refined_coords.reset_index(drop=True, inplace=True)

    # mass and signal values has to be corrected due to the rescaling
    # raw_mass was obtained from raw image; size and ecc are scale-independent
    refined_coords['mass'] /= scale_factor
    if 'signal' in refined_coords:
        refined_coords['signal'] /= scale_factor

    # Filter on mass and size, if set.
    condition = refined_coords['mass'] > minmass
    if maxsize is not None:
        condition &= refined_coords['size'] < maxsize
    if not condition.all():  # apply the filter
        # making a copy to avoid SettingWithCopyWarning
        refined_coords = refined_coords.loc[condition].copy()

    if len(refined_coords) == 0:
        warnings.warn("No maxima survived mass- and size-based filtering. "
                      "Be advised that the mass computation was changed from "
                      "version 0.2.4 to 0.3.0 and from 0.3.3 to 0.4.0. "
                      "See the documentation and the convenience functions "
                      "'minmass_v03_change' and 'minmass_v04_change'.")
        return refined_coords

    if topn is not None and len(refined_coords) > topn:
        # go through numpy for easy pandas backwards compatibility
        mass = refined_coords['mass'].values
        if topn == 1:
            # special case for high performance and correct shape
            refined_coords = refined_coords.iloc[[np.argmax(mass)]]
        else:
            refined_coords = refined_coords.iloc[np.argsort(mass)[-topn:]]

    # Estimate the uncertainty in position using signal (measured in refine)
    # and noise (measured here below).
    if characterize:
        black_level, noise = measure_noise(image, raw_image, radius)
        Npx = N_binary_mask(radius, ndim)
        mass = refined_coords['raw_mass'].values - Npx * black_level
        ep = _static_error(mass, noise, radius, noise_size)

        if ep.ndim == 1:
            refined_coords['ep'] = ep
        else:
            ep = pd.DataFrame(ep, columns=['ep_' + cc for cc in pos_columns])
            refined_coords = pandas_concat([refined_coords, ep], axis=1)

    # If this is a pims Frame object, it has a frame number.
    # Tag it on; this is helpful for parallelization.
    if hasattr(raw_image, 'frame_no') and raw_image.frame_no is not None:
        refined_coords['frame'] = int(raw_image.frame_no)
    return refined_coords


def batch(frames, diameter, output=None, meta=None, processes='auto',
          after_locate=None, **kwargs):
    """Locate Gaussian-like blobs of some approximate size in a set of images.

    Preprocess the image by performing a band pass and a threshold.
    Locate all peaks of brightness, characterize the neighborhoods of the peaks
    and take only those with given total brightness ("mass"). Finally,
    refine the positions of each peak.

    Parameters
    ----------
    frames : list (or iterable) of images
        The frames to process.
    diameter : odd integer or tuple of odd integers
        This may be a single number or a tuple giving the feature's
        extent in each dimension, useful when the dimensions do not have
        equal resolution (e.g. confocal microscopy). The tuple order is the
        same as the image shape, conventionally (z, y, x) or (y, x). The
        number(s) must be odd integers. When in doubt, round up.
    output : {None, trackpy.PandasHDFStore, SomeCustomClass}
        If None, return all results as one big DataFrame. Otherwise, pass
        results from each frame, one at a time, to the put() method
        of whatever class is specified here.
    meta : filepath or file object
        If specified, information relevant to reproducing this batch is saved
        as a YAML file, a plain-text machine- and human-readable format.
        By default, this is None, and no file is saved.
    processes : integer or "auto"
        The number of processes to use in parallel. If <= 1, multiprocessing is
        disabled. If "auto", the number returned by `os.cpu_count()`` is used.
    after_locate : function
        Specify a custom function to apply to the detected features in each
        processed frame. It must accept the following arguments:

        - ``frame_no``: an integer specifying the number of the current frame.
        - ``features``: a DataFrame containing the detected features.

        Furthermore it must return a DataFrame like ``features``.
    **kwargs :
        Keyword arguments that are passed to the wrapped `trackpy.locate`.
        Refer to its docstring for further details.

    Returns
    -------
    DataFrame([x, y, mass, size, ecc, signal])
        where mass means total integrated brightness of the blob,
        size means the radius of gyration of its Gaussian-like profile,
        and ecc is its eccentricity (0 is circular).

    See Also
    --------
    locate : performs location on a single image

    Notes
    -----
    This is a convenience function that wraps `trackpy.locate` (see its
    docstring for further details) and allows batch processing of multiple
    frames, optionally in parallel by using multiprocessing.
    """
    if "raw_image" in kwargs:
        raise KeyError("the argument `raw_image` musn't be in `kwargs`, it is "
                       "provided internally by `frames`")
    # Add required keyword argument
    kwargs["diameter"] = diameter

    if meta:
        # Gather meta information and save as YAML in current directory.
        try:
            source = frames.filename
        except AttributeError:
            source = None
        meta_info = dict(
            timestamp=pd.datetime.utcnow().strftime('%Y-%m-%d-%H%M%S'),
            trackpy_version=trackpy.__version__,
            source=source,
            **kwargs
        )
        if isinstance(meta, str):
            with open(meta, 'w') as file_obj:
                record_meta(meta_info, file_obj)
        else:
            # Interpret meta to be a file handle.
            record_meta(meta_info, meta)

    # Prepare wrapped function for mapping to `frames`
    curried_locate = partial(locate, **kwargs)

    pool, map_func = get_pool(processes)

    if after_locate is None:
        def after_locate(frame_no, features):
            return features

    try:
        all_features = []
        for i, features in enumerate(map_func(curried_locate, frames)):
            image = frames[i]
            if hasattr(image, 'frame_no') and image.frame_no is not None:
                frame_no = image.frame_no
                # Even if this worked, if locate() was running in parallel,
                # it may not have had access to the "frame_no" attribute.
                # Therefore we'll add the frame number to the DataFrame if
                # needed, below.
            else:
                frame_no = i
            if 'frame' not in features.columns:
                features['frame'] = frame_no  # just counting iterations
            features = after_locate(frame_no, features)

            logger.info("Frame %d: %d features", frame_no, len(features))
            if len(features) > 0:
                # Store if features were found
                if output is None:
                    all_features.append(features)
                else:
                    output.put(features)
    finally:
        if pool:
            # Ensure correct termination of Pool
            pool.terminate()

    if output is None:
        if len(all_features) > 0:
            return pandas_concat(all_features).reset_index(drop=True)
        else:  # return empty DataFrame
            warnings.warn("No maxima found in any frame.")
            return pd.DataFrame(columns=list(features.columns) + ['frame'])
    else:
        return output


def characterize(coords, image, radius, scale_factor=1.):
    """ Characterize a 2d ndarray of coordinates. Returns a dictionary of 1d
    ndarrays. If the feature region (partly) falls out of the image, then the
    corresponding element in the characterized arrays will be NaN."""
    shape = image.shape
    N, ndim = coords.shape

    radius = validate_tuple(radius, ndim)
    isotropic = is_isotropic(radius)

    # largely based on trackpy.refine.center_of_mass._refine
    coords_i = np.round(coords).astype(int)
    mass = np.full(N, np.nan)
    signal = np.full(N, np.nan)
    ecc = np.full(N, np.nan)

    mask = binary_mask(radius, ndim).astype(np.uint8)
    if isotropic:
        Rg = np.full(len(coords), np.nan)
    else:
        Rg = np.full((len(coords), len(radius)), np.nan)

    for feat, coord in enumerate(coords_i):
        if np.any([c - r < 0 or c + r >= sh
                   for c, r, sh in zip(coord, radius, shape)]):
            continue
        rect = [slice(c - r, c + r + 1) for c, r in zip(coord, radius)]
        neighborhood = mask * image[tuple(rect)]
        mass[feat] = neighborhood.sum() / scale_factor
        signal[feat] = neighborhood.max() / scale_factor
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

    result = dict(mass=mass, signal=signal, ecc=ecc)
    if isotropic:
        result['size'] = Rg
    else:
        for _size, key in zip(Rg.T, default_size_columns(ndim, isotropic)):
            result[key] = _size
    return result
