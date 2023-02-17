import warnings
import logging

import numpy as np
from scipy import ndimage

from ..masks import slice_image, mask_image
from ..find import grey_dilation, drop_close
from ..utils import (default_pos_columns, is_isotropic, validate_tuple,
                     pandas_concat)
from ..preprocessing import bandpass
from ..refine import refine_com_arr
from ..feature import characterize

from .utils import points_from_arr
from .subnet import Subnets
from .linking import Linker

logger = logging.getLogger(__name__)


def find_link(reader, search_range, separation, diameter=None, memory=0,
              minmass=0, noise_size=1, smoothing_size=None, threshold=None,
              percentile=64, preprocess=True, before_link=None,
              after_link=None, refine=False, **kwargs):
    """Find and link features, using image data to re-find lost features.

    Parameters
    ----------
    reader : pims.FramesSequence
    search_range : number or tuple
        maximum displacement of features between subsequent frames
    separation : number or tuple
        minimum separation distance between features
    diameter : number or tuple, optional
        feature diameter, used for characterization only.
        Also determines the margin (margin = diameter // 2).
        Default: ``separation``.
    memory : number, optional
        number of frames that features are allowed to disappear. Experimental.
        Default 0.
    minmass : number, optional
        minimum integrated intensity (in masked image). Default 0.
    noise_size : number or tuple, optional
        Size of Gaussian kernel with which the image is convoluted for noise
        reduction. Default 1.
    smoothing_size : number or tuple, optional
        Size of rolling average box for background subtraction.
        By default, equals ``separation``. This may introduce bias when refined on
        the background subtracted image!
    threshold : number, optional
        Threshold value for image. Default None.
    percentile : number, optional
        The upper percentile of intensities in the image are considered as
        feature locations. Default 64.
    preprocess : boolean
        Set to False to turn off bandpass preprocessing.
    before_link : function, optional
        This function is executed after the initial find of each frame, but
        but before the linking and relocating.
        It should take the following arguments (or ``**kwargs``):

        - ``coords``: `ndarray``containing the initially found feature coordinates
        - ``reader``: unprocessed reader (for access to other frames)
        - ``image``: unprocessed image
        - ``image_proc``: the processed image
        - ``diameter``
        - ``separation``
        - ``search_range``
        - ``margin``
        - ``minmass``

        It should return an ndarray of the same shape as ``coords``.
    after_link : function, optional
        This function is executed after the find and link of each frame. It
        should not change the number of features.
        It should take the following arguments (or ``**kwargs``):

        - ``features``: a DataFrame containing the feature coordinates and characterization.
        - ``reader``: unprocessed reader (for access to other frames)
        - ``image``: unprocessed image
        - ``image_proc``: the processed image
        - ``diameter``
        - ``separation``
        - ``search_range``
        - ``margin``
        - ``minmass``

        It should return a DataFrame like ``features``.
    refine : boolean, optional
        Convenience parameter to do center-of-mass refinement. Cannot be used
        combined with an ``after_link`` function. Default False.

    Notes
    -----
    This feature is a recent addition to trackpy that is still in its
    experimental phase. Please report any issues you encounter on Github.

    If you use this specific algorithm for your scientific publications, please
    mention the accompanying publication [1]_

    References
    ----------
    .. [1] van der Wel C., Kraft D.J. Automated tracking of colloidal clusters
    with sub-pixel accuracy and precision. J. Phys. Condens. Mat. 29:44001 (2017)
    DOI: http://dx.doi.org/10.1088/1361-648X/29/4/044001
    """
    shape = reader[0].shape
    ndim = len(shape)
    separation = validate_tuple(separation, ndim)
    if diameter is None:
        diameter = separation
    else:
        diameter = validate_tuple(diameter, ndim)

    if preprocess:
        if smoothing_size is None:
            smoothing_size = separation
        smoothing_size = validate_tuple(smoothing_size, ndim)
        # make smoothing_size an odd integer
        smoothing_size = tuple([int((s - 1) / 2) * 2 + 1 for s in smoothing_size])
        proc_func = lambda x: bandpass(x, noise_size, smoothing_size,
                                       threshold)
    else:
        proc_func = None

    if refine:
        if after_link is not None:
            raise ValueError("Refine cannot be used together with after_link.")
        pos_columns = default_pos_columns(ndim)
        refine_columns = pos_columns[::-1] + ['mass']
        radius = tuple([d // 2 for d in diameter])
        def after_link(image, features, image_proc, **kwargs):
            coords = features[pos_columns].values
            if len(coords) == 0:
                return features
            # no separation filtering, because we use precise grey dilation
            coords = refine_com_arr(image, image_proc, radius, coords, separation=0,
                                    characterize=False)
            features[refine_columns] = coords
            return features

    features = []
    generator = find_link_iter(reader, search_range, separation,
                               diameter=diameter, memory=memory,
                               percentile=percentile, minmass=minmass,
                               proc_func=proc_func, before_link=before_link,
                               after_link=after_link, **kwargs)
    for frame_no, f_frame in generator:
        if f_frame is None:
            n_traj = 0
        else:
            n_traj = len(f_frame)
        logger.info("Frame {}: {} trajectories present.".format(frame_no,
                                                                  n_traj))
        if n_traj == 0:
            continue
        features.append(f_frame)

    features = pandas_concat(features, ignore_index=False)
    return features


def find_link_iter(reader, search_range, separation, diameter=None,
                   percentile=64, minmass=0, proc_func=None, before_link=None,
                   after_link=None, **kwargs):

    shape = reader[0].shape
    ndim = len(shape)

    search_range = validate_tuple(search_range, ndim)
    separation = validate_tuple(separation, ndim)
    isotropic = is_isotropic(diameter)
    if proc_func is None:
        proc_func = lambda x: x

    if diameter is None:
        diameter = separation
    else:
        diameter = validate_tuple(diameter, ndim)
    radius = tuple([int(d // 2) for d in diameter])
    # Define zone of exclusion at edges of image, avoiding features with
    # incomplete image data ("radius")
    margin = radius

    # Check whether the margins are not covering the complete image
    if np.any([s <= 2*m for (s, m) in zip(shape, margin)]):
        # Check whether the image looks suspiciously like a multichannel image.
        if np.any([s <= 4 for s in shape]) and (ndim > 2):
            raise ValueError('One of the image dimensions is very small. '
                             'Please make sure that you are not using an RGB '
                             'or other multichannel (color) image.')
        else:
            raise ValueError('The feature finding margins are larger than the '
                             'image shape. Please use smaller radius, '
                             'separation or smoothing_size.')

    linker = FindLinker(search_range, separation, diameter, minmass,
                        percentile, **kwargs)

    reader_iter = iter(reader)
    image = next(reader_iter)
    image_proc = proc_func(image)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coords = grey_dilation(image_proc, separation, percentile, margin,
                               precise=True)
    if before_link is not None:
        coords = before_link(coords=coords, reader=reader, image=image,
                             image_proc=image_proc,
                             diameter=diameter, separation=separation,
                             search_range=search_range,
                             margin=margin, minmass=minmass)
    extra_data = characterize(coords, image, radius)
    mask = extra_data['mass'] >= minmass
    coords = coords[mask]
    for key in extra_data:
        extra_data[key] = extra_data[key][mask]
    linker.init_level(coords, image.frame_no, extra_data)
    features = linker.coords_df
    if after_link is not None and features is not None:
        features = after_link(features=features, reader=reader, image=image,
                              image_proc=image_proc,
                              diameter=diameter, separation=separation,
                              search_range=search_range, margin=margin,
                              minmass=minmass)
        linker.coords_df = features  # for next iteration

    yield image.frame_no, features

    for image in reader_iter:
        image_proc = proc_func(image)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coords = grey_dilation(image_proc, separation, percentile, margin,
                                   precise=True)
        if before_link is not None:
            coords = before_link(coords=coords, reader=reader, image=image,
                                 image_proc=image_proc,
                                 diameter=diameter, separation=separation,
                                 search_range=search_range,
                                 margin=margin, minmass=minmass)
        extra_data = characterize(coords, image, radius)
        mask = extra_data['mass'] >= minmass
        coords = coords[mask]
        for key in extra_data:
            extra_data[key] = extra_data[key][mask]
        linker.next_level(coords, image.frame_no, image=image_proc,
                          extra_data=extra_data)
        features = linker.coords_df
        if after_link is not None and features is not None:
            features = after_link(features=features, reader=reader, image=image,
                                  image_proc=image_proc,
                                  diameter=diameter, separation=separation,
                                  search_range=search_range, margin=margin,
                                  minmass=minmass)
            linker.coords_df = features  # for next iteration
        yield image.frame_no, features


class FindLinker(Linker):
    """ Linker that uses image data to re-find lost features.

    Newly found features are farther than ``separation`` from any other feature
    in the current frame, closer than ``search_range`` to a feature in the
    previous frame, and have minimum integrated intensity ``minmass`` in the
    feature region (defined by ``diameter``).

    Parameters
    ----------
    search_range : tuple
        The maximum distance features can move between frames, in pixels.
    separation : tuple
        The minimum distance between features, in pixels.
    diameter : tuple
        Size used in the characterization of new features.
        Also determines the margin (margin = diameter // 2).
    memory : int, optional
        Default 0
    minmass : number, optional
        Minimum summed intensity (in the masked image) of relocated features.
        Default 0.
    percentile : number, optional
        Precentile threshold used in local maxima finding. Default 64.


    Methods
    -------
    next_level(coords, t, image, extra_data)
        Link and relocate the next frame, using the extra parameter ``image``.
    relocate(source_points, n)
        Relocate ``n`` points close to source_points
    get_relocate_candidates(source_points)
        Obtain relacote coordinates of new features close to ``source_points``

    See also
    --------
    Linker
    """
    def __init__(self, search_range, separation, diameter=None,
                 minmass=0, percentile=64, **kwargs):
        if 'dist_func' in kwargs:
            warnings.warn("Custom distance functions are untested using "
                          "the FindLinker and likely will cause issues!")
        # initialize the Linker.
        # beware: self.search_range is a scalar, while search_range is a tuple
        super().__init__(search_range, **kwargs)
        self.ndim = len(search_range)
        if diameter is None:
            diameter = separation
        self.radius = tuple([int(d // 2) for d in diameter])
        self.separation = separation
        self.minmass = minmass  # in masked image
        self.percentile = percentile

        # For grey dilation: find the largest box that fits inside the ellipse
        # given by separation
        self.dilation_size = tuple([int(2 * s / np.sqrt(self.ndim))
                                   for s in self.separation])
        # slice_radius: radius for relocate mask
        # search_range + feature radius + 1
        self.slice_radius = tuple([int(s + r + 1)
                                   for (s, r) in zip(search_range,
                                                     self.radius)])
        # background_radius: radius to make sure the already located features
        # do not fall inside slice radius
        bg_radius = [sl + r + 1 for (sl, r) in zip(self.slice_radius,
                                                   self.radius)]
        # The big feature hashtable is normed to search_range. For performance,
        # we do not rebuild this large hashtable. apply the norm here and take
        # the largest value.
        if is_isotropic(search_range):
            self.bg_radius = max(bg_radius)
        else:
            self.bg_radius = max([a / b for (a, b) in zip(bg_radius,
                                                          search_range)])
        self.threshold = (None, None)

    def next_level(self, coords, t, image, extra_data=None):
        self.image = image
        self.curr_t = t
        prev_hash = self.update_hash(coords, t, extra_data)

        self.subnets = Subnets(prev_hash, self.hash, self.search_range,
                               self.MAX_NEIGHBORS)
        spl, dpl = self.assign_links()
        self.apply_links(spl, dpl)

    def relocate(self, pos, n=1):
        candidates, extra_data = self.get_relocate_candidates(pos)
        if candidates is None:
            return set()
        else:
            n = min(n, len(candidates))
            points = points_from_arr(candidates[:n], self.curr_t, extra_data)
        return set(points)

    def percentile_threshold(self, percentile):
        frame_no, threshold = self.threshold
        if self.curr_t != frame_no:
            not_black = self.image[np.nonzero(self.image)]
            if len(not_black) == 0:
                threshold = None
            else:
                threshold = np.percentile(not_black, percentile)
            self.threshold = (self.curr_t, threshold)
        return threshold

    def get_relocate_candidates(self, pos):
        # pos are the estimated locations of the features (ndarray N x ndim)
        pos = np.atleast_2d(pos)

        # slice region around cluster
        im_unmasked, origin = slice_image(pos, self.image, self.slice_radius)

        # return when there is no intensity left
        if im_unmasked.sum() == 0:
            return None, None
        # mask image so that only regions up to slice_radius are visible
        im_masked = mask_image(pos, im_unmasked, self.slice_radius, origin,
                               invert=False)
        # return when there is no intensity left
        if im_masked.sum() == 0:
            return None, None

        # mask coords that were already found ('background')
        background = self.hash.query_points(pos, self.bg_radius)
        if background is not None:
            im_masked = mask_image(background, im_masked, self.separation,
                                   origin, invert=True)

        threshold = self.percentile_threshold(self.percentile)
        if threshold is None:  # completely black image
            return None, None
        if np.all(im_masked < threshold):  # image entirely below threshold
            return None, None
        # The intersection of the image with its dilation gives local maxima.
        dilation = ndimage.grey_dilation(im_masked, self.dilation_size,
                                         mode='constant')
        maxima = (im_masked == dilation) & (im_masked > threshold)
        if np.sum(maxima) == 0:   # no maxima
            return None, None
        coords = np.vstack(np.where(maxima)).T

        # Do not accept peaks near the edges.
        shape = np.array(self.image.shape)
        near_edge = np.any((coords < self.radius) |
                           (coords > (shape - self.radius - 1)), axis=1)
        coords = coords[~near_edge]
        if len(coords) == 0:
            return None, None

        # drop points that are further than search range from all initial points
        # borrow the rescaling function from the hash
        coords_rescaled = self.hash.to_eucl(origin + coords)
        pos_rescaled = self.hash.to_eucl(pos)
        coords_ok = []
        for coord, coord_rescaled in zip(coords, coords_rescaled):
            dists = np.sqrt(np.sum((coord_rescaled - pos_rescaled)**2, axis=1))
            if np.any(dists <= self.search_range):
                coords_ok.append(coord)
        if len(coords_ok) == 0:
            return None, None
        coords = np.array(coords_ok)

        # drop dimmer points that are closer than separation to each other
        coords = drop_close(coords, self.separation,
                            [im_masked[tuple(c)] for c in coords])
        if coords is None:
            return None, None

        try:
            scale_factor = self.image.metadata['scale_factor']
        except (AttributeError, KeyError):
            scale_factor = 1.
        extra_data = characterize(coords, im_masked, self.radius, scale_factor)

        mass = extra_data['mass']
        mask = np.argsort(mass)[::-1][:np.sum(mass >= self.minmass)]
        for key in extra_data:
            extra_data[key] = extra_data[key][mask]
        return coords[mask] + origin, extra_data

    def assign_links(self):
        # The following method includes subnets with only one source point
        self.subnets.include_lost()
        # Also, it merges subnets that are less than 2*search_range spaced,
        # to account for lost particles that link subnets together. A possible
        # performance enhancement would be joining subnets together during
        # iterating over the subnets.
        self.subnets.merge_lost_subnets(self.search_range)

        spl, dpl = [], []
        for source_set, dest_set in self.subnets:
            # relocate if necessary
            shortage = len(source_set) - len(dest_set)
            if shortage > 0:
                if self.predictor is not None:
                    # lookup the predicted locations
                    sh = self.subnets.source_hash
                    pos = [c for c, p in zip(sh.coords_mapped,
                                             sh.points) if p in source_set]
                else:
                    pos = [s.pos for s in source_set]
                new_cands = self.relocate(pos, shortage)
                # this adapts the dest_set inplace
                self.subnets.add_dest_points(source_set, new_cands,
                                             self.search_range)
            else:
                new_cands = set()

            for sp in source_set:
                sp.forward_cands.sort(key=lambda x: x[1])

            # link
            sn_spl, sn_dpl = self.subnet_linker(source_set, dest_set,
                                                self.search_range)

            # list the claimed destination particles and add them to the hash
            sn_dpl_set = set(sn_dpl)
            # claimed new destination particles
            for p in new_cands & sn_dpl_set:
                self.hash.add_point(p)
            # unclaimed old destination particles
            unclaimed = (dest_set - sn_dpl_set) - new_cands
            sn_spl.extend([None] * len(unclaimed))
            sn_dpl.extend(unclaimed)

            spl.extend(sn_spl)
            dpl.extend(sn_dpl)

        return spl, dpl
