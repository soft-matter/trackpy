import unittest

import numpy as np
from pandas import DataFrame
from numpy.testing import assert_allclose
from scipy.spatial import cKDTree

import trackpy as tp
from trackpy import Polydisperse
from trackpy.artificial import draw_feature
from trackpy.find import grey_dilation, where_close_variable
from trackpy.spiff import apply_spiff, _merged_bins, MIN_FEATURES
from trackpy.utils import default_pos_columns
from trackpy.tests.common import StrictTestCase


def draw_polydisperse(shape, features, max_value=200, noise=0, seed=0,
                      aspect=None):
    """Draw Gaussian features of given (position, radius-of-gyration).

    ``features`` is a list of ``(position, size)`` tuples, where ``size`` is the
    feature's radius of gyration (the units of :func:`draw_feature`). When
    ``aspect`` is given, each feature is an axis-aligned ellipse whose per-axis
    radius of gyration is ``size * aspect[i]``.
    """
    image = np.zeros(shape, dtype=np.uint8)
    for position, size in features:
        per_axis = size if aspect is None else tuple(size * a for a in aspect)
        draw_feature(image, position, per_axis, max_value=max_value)
    if noise:
        rng = np.random.default_rng(seed)
        image = np.clip(image.astype(float) + rng.normal(0, noise, shape),
                        0, 255).astype(np.uint8)
    return image


def saturated_blob(shape, center, sigma, amp=600):
    """A Gaussian bright enough to clip, giving a flat (plateaued) top."""
    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    r2 = sum((g - c) ** 2 for g, c in zip(grids, center))
    return np.clip(amp * np.exp(-r2 / (2.0 * sigma ** 2)), 0, 255).astype(np.uint8)


def pack_polydisperse(shape, max_diameter, gap, seed, target=120,
                      max_attempts=12000):
    """Randomly place non-overlapping features spanning a diameter range.

    Sizes are drawn so diameters span roughly [5, max_diameter]; a candidate is
    rejected if its centre is closer than the sum of the two features' (visible)
    radii plus ``gap`` to any accepted feature (so a smaller ``gap`` packs more
    densely). Returns (centers, sizes) where size = radius of gyration.
    """
    rng = np.random.default_rng(seed)
    max_rg = max_diameter / 3.5
    margin = max_diameter // 2 + 3
    low, high = margin, shape[0] - margin
    centers = np.empty((0, len(shape)))
    radii = np.empty(0)
    sizes = []
    for _ in range(max_attempts):
        if len(sizes) >= target:
            break
        radius_of_gyration = rng.uniform(1.4, max_rg)
        visible_radius = 2.0 * radius_of_gyration
        candidate = rng.uniform(low, high, len(shape))
        distances = np.hypot(*(centers - candidate).T)  # to every accepted feature
        if len(centers) == 0 or np.all(distances >= radii + visible_radius + gap):
            centers = np.vstack([centers, candidate])
            radii = np.append(radii, visible_radius)
            sizes.append(radius_of_gyration)
    return centers, np.array(sizes)


def pack_polydisperse_aniso(shape, max_ref_diameter, gap, seed, aspect,
                            target=60, max_attempts=40000):
    """Place non-overlapping fixed-shape ellipsoids spanning a size range.

    Like :func:`pack_polydisperse` but N-dimensional and anisotropic: every
    feature is an axis-aligned ellipsoid whose per-axis radius is scaled by
    ``aspect``. Rejection is done in aspect-normalised coordinates -- where the
    ellipsoids become spheres of the reference ``visible_radius`` -- so features
    never overlap regardless of orientation. The per-axis placement margin also
    follows ``aspect`` (a thin z axis gets a thin margin). Returns (centers,
    sizes) with size = reference radius of gyration (the aspect==1 axes' value).
    """
    rng = np.random.default_rng(seed)
    max_rg = max_ref_diameter / 3.5
    ndim = len(shape)
    asp = np.array(aspect, dtype=float)
    margin = [int(max_ref_diameter * a) // 2 + 3 for a in aspect]
    centers = np.empty((0, ndim))
    visible_radii = np.empty(0)
    sizes = []
    for _ in range(max_attempts):
        if len(sizes) >= target:
            break
        radius_of_gyration = rng.uniform(1.4, max_rg)
        visible_radius = 2.0 * radius_of_gyration
        candidate = np.array([rng.uniform(m, s - m)
                              for s, m in zip(shape, margin)])
        normalized = np.sqrt(np.sum(((centers - candidate) / asp) ** 2, axis=1))
        if len(centers) == 0 or np.all(
                normalized >= visible_radii + visible_radius + gap):
            centers = np.vstack([centers, candidate])
            visible_radii = np.append(visible_radii, visible_radius)
            sizes.append(radius_of_gyration)
    return centers, np.array(sizes)


def _recall_fraction(features, true_centers, tolerance):
    """Fraction of true centers with a detection within ``tolerance`` pixels."""
    true_centers = np.asarray(true_centers)
    if len(features) == 0:
        return 0.0
    cols = default_pos_columns(true_centers.shape[1])
    distance_to_detection, _ = cKDTree(features[cols].values).query(true_centers)
    return float(np.mean(distance_to_detection < tolerance))


class TestPolydisperseConfig(StrictTestCase):
    def test_valid_construction(self):
        p = Polydisperse(5, 21)
        self.assertEqual(p.min_diameter, 5)
        self.assertEqual(p.max_diameter, 21)
        self.assertEqual(p.edge_frac, 0.1)
        # Isotropic tuples are accepted.
        Polydisperse((7, 7), (11, 11))
        # edge_frac is overridable.
        self.assertEqual(Polydisperse(5, 21, edge_frac=0.2).edge_frac, 0.2)

    def test_for_ndim(self):
        sizes = Polydisperse(5, 21).for_ndim(2)
        self.assertEqual(sizes.min_diameter, (5, 5))
        self.assertEqual(sizes.max_diameter, (21, 21))
        self.assertEqual(sizes.r_max, (10, 10))
        self.assertEqual(sizes.aspect, (1.0, 1.0))

    def test_invalid_construction(self):
        for args in [(4, 21), (5, 20), (21, 5), ((5, 7), (11, 11)), (5.0, 21)]:
            with self.assertRaises(ValueError):
                Polydisperse(*args)
        for bad_edge in [0, 1, -0.1, 1.5]:
            with self.assertRaises(ValueError):
                Polydisperse(5, 21, edge_frac=bad_edge)

    def test_default_aspect_is_isotropic(self):
        self.assertEqual(Polydisperse(5, 21).aspect, 1)

    def test_aspect_stored_and_scaled_per_axis(self):
        p = Polydisperse(5, 21, aspect=(1, 2))
        self.assertEqual(p.aspect, (1, 2))
        sizes = p.for_ndim(2)
        self.assertEqual(sizes.aspect, (1.0, 2.0))
        self.assertEqual(sizes.min_diameter, (5, 9))
        self.assertEqual(sizes.max_diameter, (21, 41))
        self.assertEqual(sizes.r_max, (10, 20))

    def test_aspect_applied_as_raw_multiplier(self):
        # aspect is used as-is (not normalised): the reference scale maps onto the
        # aspect==1 axes (e.g. confocal x/y), the others scale by their ratio.
        sizes = Polydisperse(9, 21, aspect=(1 / 3, 1, 1)).for_ndim(3)
        self.assertEqual(sizes.aspect[1:], (1.0, 1.0))
        self.assertAlmostEqual(sizes.aspect[0], 1 / 3)
        self.assertEqual((sizes.ref_min, sizes.ref_max), (9, 21))
        # x/y (aspect 1) carry the reference diameters; z (axis 0) is compressed.
        self.assertEqual(sizes.min_diameter[1:], (9, 9))
        self.assertEqual(sizes.max_diameter[1:], (21, 21))
        self.assertLess(sizes.max_diameter[0], sizes.max_diameter[1])

    def test_invalid_aspect(self):
        for bad in [0, -1, (1, 0), (1, -2), ()]:
            with self.assertRaises(ValueError):
                Polydisperse(5, 21, aspect=bad)


class TestPolydisperseLocate(StrictTestCase):
    def test_assigns_larger_window_to_larger_feature(self):
        image = draw_polydisperse((120, 120),
                                  [((35, 35), 1.5), ((80, 80), 4.0)])
        f = tp.locate(image, Polydisperse(5, 21))
        self.assertEqual(len(f), 2)
        self.assertIn('diameter', f.columns)
        self.assertTrue((f['diameter'] % 2 == 1).all())
        self.assertTrue(f['diameter'].between(5, 21).all())
        small = f.loc[((f.y - 35) ** 2 + (f.x - 35) ** 2).idxmin()]
        large = f.loc[((f.y - 80) ** 2 + (f.x - 80) ** 2).idxmin()]
        self.assertGreater(large['diameter'], small['diameter'])
        assert_allclose([small.y, small.x], [35, 35], atol=0.5)
        assert_allclose([large.y, large.x], [80, 80], atol=0.5)

    def test_single_feature_gives_single_detection(self):
        # Detection (flat-peak collapse) plus dedup yield one feature, not many.
        image = draw_polydisperse((80, 80), [((40, 40), 4.0)])
        f = tp.locate(image, Polydisperse(5, 21))
        self.assertEqual(len(f), 1)

    def test_oversized_feature_clamped_to_max(self):
        # A feature larger than max_diameter is not dropped; it is refined at
        # (clamped to) the maximum diameter and still reported.
        image = draw_polydisperse((140, 140),
                                  [((40, 40), 2.5), ((100, 100), 10.0)])
        f = tp.locate(image, Polydisperse(5, 11))
        self.assertTrue((f['diameter'] <= 11).all())
        big = f.loc[((f.y - 100) ** 2 + (f.x - 100) ** 2).idxmin()]
        assert_allclose([big.y, big.x], [100, 100], atol=1.0)
        self.assertEqual(big['diameter'], 11)

    def test_separate_small_features_not_merged(self):
        image = draw_polydisperse((80, 80),
                                  [((30, 25), 1.5), ((30, 50), 1.5)])
        f = tp.locate(image, Polydisperse(5, 21))
        self.assertEqual(len(f), 2)

    def test_ep_present_and_finite(self):
        image = draw_polydisperse((160, 160),
                                  [((45, 45), 1.5), ((110, 110), 4.0)],
                                  noise=2.0)
        f = tp.locate(image, Polydisperse(5, 21), minmass=200)
        self.assertIn('ep', f.columns)
        self.assertTrue(np.isfinite(f['ep']).all())
        self.assertTrue((f['ep'] > 0).all())

    def test_batch(self):
        frames = [draw_polydisperse((90, 90),
                                    [((30, 30), 1.5), ((60, 60), 3.5)])
                  for _ in range(2)]
        b = tp.batch(frames, Polydisperse(5, 21), processes=1)
        self.assertIn('diameter', b.columns)
        self.assertEqual(sorted(b['frame'].unique().tolist()), [0, 1])

    def test_monodisperse_unaffected(self):
        image = draw_polydisperse((60, 60), [((30, 30), 2.0)])
        f = tp.locate(image, 7)
        self.assertNotIn('diameter', f.columns)
        self.assertEqual(len(f), 1)

    def test_small_near_large_not_deduplicated(self):
        # A small particle just outside a large particle's body must survive:
        # dedup should key on each feature's radius, not its full separation,
        # or the large particle's separation would delete the small neighbour.
        image = draw_polydisperse((100, 100), [((50, 50), 5.0), ((50, 66), 1.5)])
        f = tp.locate(image, Polydisperse(5, 21))
        self.assertEqual(len(f), 2)
        small = f.loc[((f.y - 50) ** 2 + (f.x - 66) ** 2).idxmin()]
        large = f.loc[((f.y - 50) ** 2 + (f.x - 50) ** 2).idxmin()]
        assert_allclose([small.y, small.x], [50, 66], atol=1.0)
        self.assertGreater(large['diameter'], small['diameter'])


class TestPolydisperseDetection(StrictTestCase):
    def test_collapse_flat_reduces_plateau_to_one(self):
        image = saturated_blob((60, 60), (30, 30), sigma=5)
        kw = dict(separation=(11, 11), margin=(10, 10), precise=False)
        raw = grey_dilation(image, collapse_flat=False, **kw)
        collapsed = grey_dilation(image, collapse_flat=True, **kw)
        self.assertGreater(len(raw), 1)
        self.assertEqual(len(collapsed), 1)
        assert_allclose(collapsed[0], [30, 30], atol=1)

    def test_collapse_keeps_distinct_peaks(self):
        image = np.zeros((60, 60), dtype=np.uint8)
        draw_feature(image, (30, 20), 2.0, max_value=200)
        draw_feature(image, (30, 45), 2.0, max_value=200)
        collapsed = grey_dilation(image, (7, 7), margin=(5, 5), precise=False,
                                  collapse_flat=True)
        self.assertEqual(len(collapsed), 2)


class TestWhereCloseVariable(StrictTestCase):

    def test_drops_dimmer_of_close_pair(self):
        self.assertEqual(
            list(where_close_variable([[0, 0], [0, 3]], [11, 11], [10, 5])), [1])

    def test_keeps_features_beyond_separation(self):
        self.assertEqual(
            list(where_close_variable([[0, 0], [0, 8]], [5, 5], [1, 1])), [])

    def test_small_inside_large_zone_is_dropped(self):
        # feature 0 is small (sep 3) but sits within feature 1's zone (sep 15)
        self.assertEqual(
            list(where_close_variable([[0, 0], [0, 4]], [3, 15], [2, 100])), [0])

    def test_intensity_tie_is_deterministic(self):
        self.assertEqual(
            list(where_close_variable([[0, 0], [0, 3]], [11, 11], [5, 5])), [1])

    def test_empty(self):
        self.assertEqual(list(where_close_variable(np.empty((0, 2)), [], [])), [])

    def test_aspect_stretches_zone_along_axis(self):
        # aspect (1, 3): axis-1 distances count for 1/3, so points 9 apart in x
        # are only 3 apart in normalised space and fall inside separation 4.
        pos = [[0, 0], [0, 9]]
        self.assertEqual(
            list(where_close_variable(pos, [4, 4], [10, 5], aspect=(1, 3))), [1])
        # Isotropically (no aspect) they are 9 apart -> not a duplicate.
        self.assertEqual(list(where_close_variable(pos, [4, 4], [10, 5])), [])

    def test_aspect_does_not_stretch_perpendicular_axis(self):
        # The same 9-px gap along the un-stretched axis 0 keeps them distinct.
        pos = [[0, 0], [9, 0]]
        self.assertEqual(
            list(where_close_variable(pos, [4, 4], [10, 5], aspect=(1, 3))), [])


class TestPolydisperseSpiff(StrictTestCase):
    def test_merged_bins(self):
        self.assertEqual(len(_merged_bins(np.array([5] * 60 + [7] * 60), 50)), 2)
        self.assertEqual(
            len(_merged_bins(np.array([5] * 10 + [7] * 10 + [9] * 100), 50)), 1)
        merged = _merged_bins(np.array([5] * 60 + [7] * 10), 50)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].sum(), 70)

    @staticmethod
    def _nonuniformity(frac):
        counts, _ = np.histogram(frac % 1, bins=10, range=(0, 1))
        return counts.std() / counts.mean()

    def test_size_class_aware_beats_pooled(self):
        rng = np.random.default_rng(0)
        n = 400
        # Class A is strongly pixel-locked (U-shaped); class B is ~uniform.
        x_a = rng.integers(20, 80, n) + rng.beta(0.35, 0.35, n)
        x_b = rng.integers(20, 80, n) + rng.uniform(0, 1, n)
        df = DataFrame({'x': np.concatenate([x_a, x_b]),
                        'y': rng.uniform(20, 80, 2 * n),
                        'diameter': np.array([5] * n + [15] * n)})
        is_a = df['diameter'].values == 5
        pooled = apply_spiff(df, groupby=None)
        grouped = apply_spiff(df, groupby='auto')
        self.assertLess(self._nonuniformity(grouped['x'].values[is_a]),
                        self._nonuniformity(pooled['x'].values[is_a]))
        self.assertLess(self._nonuniformity(grouped['x'].values[is_a]), 0.25)

    def test_auto_pools_without_diameter_column(self):
        rng = np.random.default_rng(1)
        x = rng.integers(10, 90, 200) + rng.beta(0.4, 0.4, 200)
        df = DataFrame({'x': x, 'y': rng.uniform(10, 90, 200)})
        assert_allclose(apply_spiff(df, groupby='auto')['x'],
                        apply_spiff(df, groupby=None)['x'])

    def test_locate_spiff_auto_runs(self):
        image = draw_polydisperse((80, 80),
                                  [((30, 30), 1.5), ((55, 55), 3.0)])
        f = tp.locate(image, Polydisperse(5, 21), spiff='auto')
        self.assertIn('diameter', f.columns)


class TestPolydisperseAnisotropic(StrictTestCase):
    """Fixed-aspect-ratio anisotropy: every feature is the same axis-aligned
    ellipse, only its scale varies."""

    @staticmethod
    def _draw(shape, features, max_value=200):
        # features: list of (position, (size_axis0, size_axis1[, ...])) tuples,
        # where each per-axis size is that axis' radius of gyration.
        image = np.zeros(shape, dtype=np.uint8)
        for position, size in features:
            draw_feature(image, position, size, max_value=max_value)
        return image

    def test_locates_elongated_features(self):
        # Two blobs stretched 2:1 in x, at two scales.
        image = self._draw((160, 200),
                           [((40, 50), (2.0, 4.0)), ((110, 140), (4.5, 9.0))])
        f = tp.locate(image, Polydisperse(5, 31, aspect=(1, 2)))
        self.assertEqual(len(f), 2)
        self.assertIn('diameter', f.columns)
        self.assertTrue((f['diameter'] % 2 == 1).all())
        small = f.loc[((f.y - 40) ** 2 + (f.x - 50) ** 2).idxmin()]
        large = f.loc[((f.y - 110) ** 2 + (f.x - 140) ** 2).idxmin()]
        assert_allclose([small.y, small.x], [40, 50], atol=0.7)
        assert_allclose([large.y, large.x], [110, 140], atol=0.7)
        self.assertGreater(large['diameter'], small['diameter'])

    def test_elliptical_window_recovers_shape(self):
        # The per-axis refinement window follows aspect, so the measured x-extent
        # tracks the 2:1 stretch of the feature.
        image = self._draw((160, 200), [((80, 100), (4.0, 8.0))])
        f = tp.locate(image, Polydisperse(5, 31, aspect=(1, 2)))
        self.assertEqual(len(f), 1)
        self.assertIn('size_x', f.columns)
        self.assertGreater(f['size_x'].iloc[0], 1.4 * f['size_y'].iloc[0])

    def test_isotropic_aspect_matches_default(self):
        # aspect=(1, 1) must reproduce the default isotropic path byte-for-byte.
        image = draw_polydisperse((120, 120),
                                  [((35, 35), 1.6), ((80, 80), 4.0)])
        default = tp.locate(image, Polydisperse(5, 21))
        explicit = tp.locate(image, Polydisperse(5, 21, aspect=(1, 1)))
        self.assertTrue(default.equals(explicit))

    def test_maxsize_rejected_for_anisotropic(self):
        image = self._draw((80, 80), [((40, 40), (2.0, 4.0))])
        with self.assertRaises(ValueError):
            tp.locate(image, Polydisperse(5, 21, aspect=(1, 2)), maxsize=5)

    def test_3d_anisotropic(self):
        image = np.zeros((60, 60, 60), dtype=np.uint8)
        draw_feature(image, (30, 30, 30), (2.5, 2.5, 5.0), max_value=200)
        f = tp.locate(image, Polydisperse(5, 21, aspect=(1, 1, 2)))
        self.assertGreaterEqual(len(f), 1)
        self.assertIn('diameter', f.columns)
        nearest = f.loc[((f.z - 30) ** 2 + (f.y - 30) ** 2
                         + (f.x - 30) ** 2).idxmin()]
        assert_allclose([nearest.z, nearest.y, nearest.x], [30, 30, 30], atol=1.5)


class TestPolydisperseRecallVsBaseline(StrictTestCase):
    """Poly must recover at least as many features as the naive `diameter=max`
    baseline, across a grid of size range x density x noise."""

    SHAPE = (256, 256)
    MIN_DIAMETER = 5
    MAX_DIAMETER_BY_RANGE = {'x3': 15, 'x5': 25, 'x10': 51}  # range label -> max diameter
    GAP_BY_DENSITY = {'low': 20, 'med': 9, 'high': 3}  # density label -> edge gap (smaller = denser)
    STD_BY_NOISE = {'low': 1, 'med': 5, 'high': 12}  # noise label -> background std
    MATCH_TOLERANCE = 1.5  # px between a detection and its true position
    RECALL_SLACK = 0.05  # absorbs ~single-particle noise
    MIN_RECALL = 0.95  # poly floor in every scenario

    def _recall(self, features, true_centers):
        return _recall_fraction(features, true_centers, self.MATCH_TOLERANCE)

    def test_recall_at_least_baseline_over_grid(self):
        for range_i, (range_label, max_diameter) in enumerate(
                self.MAX_DIAMETER_BY_RANGE.items()):
            for density_i, (density_label, gap) in enumerate(
                    self.GAP_BY_DENSITY.items()):
                for noise_i, (noise_label, noise_std) in enumerate(
                        self.STD_BY_NOISE.items()):
                    seed = 100 * range_i + 10 * density_i + noise_i
                    true_centers, sizes = pack_polydisperse(
                        self.SHAPE, max_diameter, gap, seed)
                    image = draw_polydisperse(
                        self.SHAPE, list(zip(true_centers, sizes)),
                        noise=noise_std, seed=seed)
                    # characterize=False: the recall/position comparison does not
                    # need the uncertainty (ep) step, which only adds cost.
                    baseline = tp.locate(image, max_diameter, minmass=80,
                                         characterize=False)
                    poly = tp.locate(
                        image, Polydisperse(self.MIN_DIAMETER, max_diameter),
                        minmass=80, characterize=False)
                    baseline_recall = self._recall(baseline, true_centers)
                    poly_recall = self._recall(poly, true_centers)
                    scenario = "%s/%s/%s (N=%d)" % (
                        range_label, density_label, noise_label, len(true_centers))
                    self.assertGreaterEqual(
                        poly_recall, baseline_recall - self.RECALL_SLACK,
                        msg="%s: poly recall %.2f < baseline %.2f"
                            % (scenario, poly_recall, baseline_recall))
                    self.assertGreaterEqual(
                        poly_recall, self.MIN_RECALL,
                        msg="%s: poly recall %.2f < floor %.2f"
                            % (scenario, poly_recall, self.MIN_RECALL))


class TestPolydisperseAnisotropicRecall(StrictTestCase):
    """Anisotropic recall in a 3-D confocal geometry: x and y equal, z one third
    as thick (aspect ``(1/3, 1, 1)``; trackpy axis order z, y, x), at the widest
    size range (x10), medium density, high noise -- the regime this feature
    targets. The naive anisotropic ``diameter=max`` baseline must use one huge
    window for every feature and loses the small, thin ones; poly's per-feature
    windows recover far more. ``minmass`` rejects the dim noise peaks (mass ~30)
    without touching the far brighter real features (mass >~150), so it costs no
    recall.
    """

    SHAPE = (30, 160, 160)  # z, y, x -- thin in z; sized so ~30+ features
    # pack (a stable base for the recall floor) while
    # the whole test still runs in well under 1 s.
    ASPECT = (1 / 3, 1, 1)
    MIN_DIAMETER = 5
    MAX_DIAMETER = 51  # x10 range on x/y
    GAP = 9  # medium density
    NOISE = 12  # high noise
    MINMASS = 50  # above the noise floor, below any real feature
    SEED = 7
    TARGET = 50
    MATCH_TOLERANCE = 1.5
    MIN_RECALL = 0.95  # poly locates ~all features (observed 1.0 here,

    # >=0.975 across seeds); floor leaves margin for a
    # ~single-feature perturbation without going flaky.

    def test_recall_beats_baseline(self):
        true_centers, sizes = pack_polydisperse_aniso(
            self.SHAPE, self.MAX_DIAMETER, self.GAP, self.SEED, self.ASPECT,
            target=self.TARGET)
        image = draw_polydisperse(
            self.SHAPE, list(zip(true_centers, sizes)),
            noise=self.NOISE, seed=self.SEED, aspect=self.ASPECT)
        # Baseline: the poly max window (per-axis) used as a single fixed size.
        max_tuple = Polydisperse(self.MIN_DIAMETER, self.MAX_DIAMETER,
                                 aspect=self.ASPECT).for_ndim(3).max_diameter
        # characterize=False skips the uncertainty (ep) step, whose measure_noise
        # is dominated by a large-radius morphological dilation in 3D and is
        # irrelevant to a recall check.
        baseline = tp.locate(image, max_tuple, minmass=self.MINMASS,
                             characterize=False)
        poly = tp.locate(image, Polydisperse(self.MIN_DIAMETER,
                                             self.MAX_DIAMETER,
                                             aspect=self.ASPECT),
                         minmass=self.MINMASS, characterize=False)
        poly_recall = _recall_fraction(poly, true_centers, self.MATCH_TOLERANCE)
        baseline_recall = _recall_fraction(baseline, true_centers,
                                           self.MATCH_TOLERANCE)
        self.assertGreater(
            poly_recall, baseline_recall + 0.3,
            msg="poly recall %.2f not far above baseline %.2f (N=%d)"
                % (poly_recall, baseline_recall, len(true_centers)))
        self.assertGreaterEqual(
            poly_recall, self.MIN_RECALL,
            msg="poly recall %.2f < floor %.2f" % (poly_recall, self.MIN_RECALL))


if __name__ == '__main__':
    unittest.main()
