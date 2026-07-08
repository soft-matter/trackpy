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


def draw_polydisperse(shape, features, max_value=200, noise=0, seed=0):
    """Draw Gaussian features of given (position, radius-of-gyration).

    ``features`` is a list of ``(position, size)`` tuples, where ``size`` is the
    feature's radius of gyration (the units of :func:`draw_feature`).
    """
    image = np.zeros(shape, dtype=np.uint8)
    for position, size in features:
        draw_feature(image, position, size, max_value=max_value)
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


def pack_polydisperse(shape, max_diameter, gap, seed, target=120, tries=12000):
    """Randomly place non-overlapping features spanning a diameter range.

    Sizes are drawn so diameters span roughly [5, max_diameter]; positions are
    rejected if closer than the sum of the two features' (visible) radii plus
    ``gap`` (so smaller ``gap`` => denser packing). Returns (positions, sizes).
    """
    rng = np.random.default_rng(seed)
    rg_hi = max_diameter / 3.5
    pad = max_diameter // 2 + 3
    lo, hi = pad, shape[0] - pad
    pos = np.empty((0, len(shape)))
    rad = np.empty(0)
    sizes = []
    for _ in range(tries):
        if len(sizes) >= target:
            break
        rg = rng.uniform(1.4, rg_hi)
        r = 2.0 * rg
        c = rng.uniform(lo, hi, len(shape))
        if len(pos) == 0 or np.all(np.hypot(*(pos - c).T) >= rad + r + gap):
            pos = np.vstack([pos, c])
            rad = np.append(rad, r)
            sizes.append(rg)
    return pos, np.array(sizes)


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

    def test_resolve(self):
        r = Polydisperse(5, 21).resolve(2)
        self.assertEqual(r.min_diameter, (5, 5))
        self.assertEqual(r.max_diameter, (21, 21))
        self.assertEqual(r.r_max, (10, 10))

    def test_invalid_construction(self):
        for args in [(4, 21), (5, 20), (21, 5), ((5, 7), (11, 11)), (5.0, 21)]:
            with self.assertRaises(ValueError):
                Polydisperse(*args)
        for bad_edge in [0, 1, -0.1, 1.5]:
            with self.assertRaises(ValueError):
                Polydisperse(5, 21, edge_frac=bad_edge)


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


class TestPolydisperseRecallVsBaseline(StrictTestCase):
    """Poly must recover at least as many features as the naive `diameter=max`
    baseline, across a grid of size range x density x noise."""

    SHAPE = (256, 256)
    MIN_D = 5
    RANGES = {'x2.5': 13, 'x5': 25, 'x10': 51}    # max_diameter
    GAPS = {'low': 20, 'med': 9, 'high': 3}       # edge gap (smaller => denser)
    NOISE = {'low': 1, 'med': 5, 'high': 12}
    TOL = 1.5                                     # match distance, px
    SLACK = 0.05                                  # absorbs ~single-particle noise
    MIN_RECALL = 0.95                             # poly floor in every scenario

    def _recall(self, features, positions):
        if len(features) == 0:
            return 0.0
        cols = default_pos_columns(2)
        d, _ = cKDTree(features[cols].values).query(positions)
        return float(np.mean(d < self.TOL))

    def test_recall_at_least_baseline_over_grid(self):
        for ri, (rname, max_d) in enumerate(self.RANGES.items()):
            for di, (dname, gap) in enumerate(self.GAPS.items()):
                for ni, (nname, noise) in enumerate(self.NOISE.items()):
                    seed = 100 * ri + 10 * di + ni
                    pos, sizes = pack_polydisperse(self.SHAPE, max_d, gap, seed)
                    image = draw_polydisperse(self.SHAPE, list(zip(pos, sizes)),
                                              noise=noise, seed=seed)
                    base = tp.locate(image, max_d, minmass=80)
                    poly = tp.locate(image, Polydisperse(self.MIN_D, max_d),
                                     minmass=80)
                    r_base = self._recall(base, pos)
                    r_poly = self._recall(poly, pos)
                    self.assertGreaterEqual(
                        r_poly, r_base - self.SLACK,
                        msg="%s/%s/%s (N=%d): poly recall %.2f < baseline %.2f"
                            % (rname, dname, nname, len(pos), r_poly, r_base))
                    self.assertGreaterEqual(
                        r_poly, self.MIN_RECALL,
                        msg="%s/%s/%s (N=%d): poly recall %.2f < floor %.2f"
                            % (rname, dname, nname, len(pos), r_poly,
                               self.MIN_RECALL))


if __name__ == '__main__':
    unittest.main()
