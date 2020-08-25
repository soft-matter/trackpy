from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
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

    radius = tuple((int(d / 2) for d in diameter))
    old_bp_size = tuple((s * 2 + 1 for s in old_smoothing_size))

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
    parameters = locals()
    locator = ParticlesLocator(**parameters)
    return locator.locate()


class ParticlesLocator:
    """Class enabling location of the particles.

    See Also
    --------
    :py:func:`trackpy.feature.locate`

    """

    def __init__(
        self,
        raw_image,
        diameter,
        minmass=None,
        maxsize=None,
        separation=None,
        noise_size=1,
        smoothing_size=None,
        threshold=None,
        invert=False,
        percentile=64,
        topn=None,
        preprocess=True,
        max_iterations=10,
        filter_before=None,
        filter_after=None,
        characterize=True,
        engine='auto',
    ):
        self.raw_image = raw_image
        self.diameter = diameter
        self.minmass = minmass
        self.maxsize = maxsize
        self.separation = separation
        self.noise_size = noise_size
        self.smoothing_size = smoothing_size
        self.threshold = threshold
        self.invert = invert
        self.percentile = percentile
        self.topn = topn
        self.preprocess = preprocess
        self.max_iterations = max_iterations
        self.filter_before = filter_before
        self.filter_after = filter_after
        self.characterize = characterize
        self.engine = engine
        self._image = self._convert_image_from_raw()

    def locate(self):
        """Locate particles."""
        coords = self._find_coords()

        funcs = [
            self._refine_coords,
            self._flat_peaks,
            self._account_for_scaling,
            self._filter_on_mass_and_size,
            self._select_topn_coords,
            self._account_for_noise,
            self._add_frame_number_if_possible,
        ]

        for f in funcs:
            coords = f(coords)
            if coords.empty:
                msg = (
                    "No maxima survived mass- and size-based filtering. "
                    "Be advised that the mass computation was changed from "
                    "version 0.2.4 to 0.3.0 and from 0.3.3 to 0.4.0. "
                    "See the documentation and the convenience functions "
                    "'minmass_v03_change' and 'minmass_v04_change'."
                )
                warnings.warn(msg)
                break
        return coords

    @property
    def raw_image(self):
        """Raw image array."""
        return self._raw_image

    @raw_image.setter
    def raw_image(self, raw_image):
        self._raw_image = raw_image
        image = np.squeeze(raw_image)
        self._is_float_image = not np.issubdtype(image.dtype, np.integer)
        self._shape = image.shape
        # self._ndim = len(self._shape)
        self._ndim = image.ndim

        if self._is_color_image(image):
            msg = (
                "I am interpreting the image as {0}-dimensional. "
                "If it is actually a {1}-dimensional color image, "
                "convert it to grayscale first."
                .format(self._ndim, self._ndim-1)
            )
            warnings.warn(msg)

    @staticmethod
    def _is_color_image(image):
        """Check if image is probably a color image."""
        shape = image.shape
        return 3 in shape or 4 in shape

    def _convert_image_from_raw(self):
        image = np.squeeze(self.raw_image)

        if self.invert:
            image = invert_image(self.raw_image)

        # Determine `image`: the image to find the local maxima on.
        if self.preprocess:
            image = bandpass(
                image, self.noise_size, self.smoothing_size, self.threshold,
            )

        # Function convert_to_int should be refactored,
        # so that it only converts to int.
        # Scale factor computation should be carried out in a separate func.
        self._scale_factor, image = convert_to_int(image, self._dtype)
        return image

    @property
    def image(self):
        """Processed image from raw image."""
        return self._image

    @property
    def minmass(self):
        return self._minmass

    @minmass.setter
    def minmass(self, value):
        self._minmass = value or 0

    @property
    def maxsize(self):
        return self._maxsize

    @maxsize.setter
    def maxsize(self, value):
        self._maxsize = value

    @property
    def diameter(self):
        return self._diameter

    @diameter.setter
    def diameter(self, diameter):
        diameter = validate_tuple(diameter, self._ndim)
        diameter = tuple([int(x) for x in diameter])
        if not np.all([x & 1 for x in diameter]):
            msg = "Feature diameter must be an odd integer. Round up."
            raise ValueError(msg)
        self._diameter = diameter
        self._radius = tuple([x//2 for x in diameter])

    @property
    def radius(self):
        return self._radius

    @property
    def invert(self):
        return self._invert

    @invert.setter
    def invert(self, value):
        if value:
            msg = (
                "The invert argument will be deprecated. "
                "Use a PIMS pipeline for this."
            )
            warnings.warn(msg, PendingDeprecationWarning)
        self._invert = value

    @property
    def separation(self):
        return self._separation

    @separation.setter
    def separation(self, value):
        if value is None:
            self._separation = tuple([x + 1 for x in self.diameter])
        else:
            self._separation = validate_tuple(value, self._ndim)

    @property
    def smoothing_size(self):
        return self._smoothing_size

    @smoothing_size.setter
    def smoothing_size(self, value):
        if value is None:
            self._smoothing_size = self.diameter
        else:
            self._smoothing_size = validate_tuple(value, self._ndim)

    @property
    def noise_size(self):
        return self._noise_size

    @noise_size.setter
    def noise_size(self, noise_size):
        self._noise_size = validate_tuple(noise_size, self._ndim)

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, threshold):
        self._threshold = threshold
        if threshold is None:
            if self._is_float_image:
                self._threshold = 1/255.
            else:
                self._threshold = 1

    @property
    def filter_before(self):
        return self._filter_before

    @filter_before.setter
    def filter_before(self, value):
        if value is not None:
            msg = (
                "The filter_before argument is no longer supported as "
                "it does not improve performance. "
                "Features are filtered after refine."
            )
            raise ValueError(msg)  # see GH issue #141
        self._filter_before = value

    @property
    def filter_after(self):
        return self._filter_after

    @filter_after.setter
    def filter_after(self, value):
        if value is not None:
            msg = (
                "The filter_after argument has been deprecated: "
                "it is always on, "
                "unless minmass = None and maxsize = None."
            )
            warnings.warn(msg, DeprecationWarning)
        self._filter_after = value

    @property
    def _dtype(self):
        # For optimal performance, performance,
        # coerce the image dtype to integer.
        if self._is_float_image:  # For float images, assume bitdepth of 8.
            return np.uint8
        else:   # For integer images, take original dtype
            return self.raw_image.dtype

    def _find_margin(self):
        """Find local maxima.

        Define zone of exclusion at edges of image, avoiding
        - Features with incomplete image data ("radius")
        - Extended particles that cannot be explored during subpixel
        refinement ("separation")
        - Invalid output of the bandpass step ("smoothing_size")
        """
        return tuple([
            max(rad, sep // 2 - 1, sm // 2)
            for (rad, sep, sm)
            in zip(self.radius, self.separation, self.smoothing_size)
        ])

    def _find_coords(self):
        """Find features with minimum separation distance of `separation`.

        This excludes detection of small features close to large,
        bright features using the `maxsize` argument.
        """
        return grey_dilation(
            self.image,
            self.separation,
            percentile=self.percentile,
            margin=self._find_margin(),
            precise=False,
        )

    def _refine_coords(self, coords):
        """Refine their locations and characterize mass, size, etc."""
        return refine_com(
            self.raw_image,
            self.image,
            self.radius,
            coords,
            max_iterations=self.max_iterations,
            engine=self.engine,
            characterize=self.characterize,
        )

    @property
    def _pos_columns(self):
        """Position column names."""
        return default_pos_columns(self._ndim)

    def _flat_peaks(self, refined_coords):
        """Flat peaks return multiple nearby maxima. Eliminate duplicates."""
        if np.all(np.greater(self.separation, 0)):
            to_drop = where_close(
                refined_coords[self._pos_columns],
                self.separation,
                refined_coords['mass'],
            )
            refined_coords.drop(to_drop, axis=0, inplace=True)
            refined_coords.reset_index(drop=True, inplace=True)
        return refined_coords

    def _account_for_scaling(self, refined_coords):
        """Correct mass and signal values due to the rescaling.

        raw_mass was obtained from raw image;
        size and ecc are scale-independent
        """
        refined_coords['mass'] /= self._scale_factor
        if 'signal' in refined_coords:
            refined_coords['signal'] /= self._scale_factor
        return refined_coords

    def _filter_on_mass_and_size(self, coords):
        """Filter coords on mass and size, if set."""
        condition = coords['mass'] > self.minmass
        if self.maxsize is not None:
            condition &= coords['size'] < self.maxsize
        if not condition.all():  # apply the filter
            # making a copy to avoid SettingWithCopyWarning
            coords = coords.loc[condition].copy()
        return coords

    def _select_topn_coords(self, coords):
        """Select top N coords."""
        if self.topn is not None and len(coords) > self.topn:
            # go through numpy for easy pandas backwards compatibility
            mass = coords['mass'].values
            if self.topn == 1:
                # special case for high performance and correct shape
                return coords.iloc[[np.argmax(mass)]]
            else:
                return coords.iloc[np.argsort(mass)[-self.topn:]]
        return coords

    def _account_for_noise(self, coords):
        """Account to uncertainty in position.

        Estimate the uncertainty in position using signal (measured in refine)
        and noise (measured here below).
        """
        if self.characterize:
            black_level, noise = measure_noise(
                image_bp=self.image,
                image_raw=self.raw_image,
                radius=self.radius,
            )

            Npx = N_binary_mask(self.radius, self._ndim)

            ep = _static_error(
                mass=coords['raw_mass'].values - Npx * black_level,
                noise=noise,
                radius=self.radius,
                noise_size=self.noise_size,
            )

            if ep.ndim == 1:
                coords['ep'] = ep
            else:
                columns = [
                    '_'.join(['ep', cc])
                    for cc in self._pos_columns
                ]
                ep = pd.DataFrame(ep, columns=columns)
                coords = pandas_concat([coords, ep], axis=1)
        return coords

    def _add_frame_number_if_possible(self, coords):
        # If this is a pims Frame object, it has a frame number.
        # Tag it on; this is helpful for parallelization.
        is_pims_frame = (
            hasattr(self.raw_image, 'frame_no')
            and self.raw_image.frame_no is not None
        )
        if is_pims_frame:
            coords['frame'] = int(self.raw_image.frame_no)
        return coords


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
        if isinstance(meta, six.string_types):
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
    coords_i = np.round(coords).astype(np.int)
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
