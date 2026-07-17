"""Benchmark: anisotropic (confocal) polydisperse ``locate`` vs. the naive
``diameter=max`` baseline, in 3-D.

Mirrors ``polydisperse_grid.py`` but in a confocal geometry: x and y equal, z
one third as thick (``aspect=(1/3, 1, 1)``; trackpy axis order z, y, x). Runs a
grid over size-range x density and reports, per cell, each method's recall
(``r``), RMS position error over matched features (``e``) and precision (``p``,
the fraction of detections that are true -- the rest are false positives), plus
grid totals and the per-method wall-clock time.

``characterize=False``: the uncertainty (ep) step is skipped. In 3-D it is
dominated by ``measure_noise``'s large-radius morphological dilation -- a cost
shared equally by both methods that would otherwise swamp the timing -- so
excluding it isolates the detection + refinement work that the poly algorithm
actually changes.

Run:  python benchmarks/polydisperse_aniso_grid.py
"""
import time

import numpy as np
from scipy.spatial import cKDTree

import trackpy as tp
from trackpy import Polydisperse
from trackpy.artificial import draw_feature
from trackpy.utils import default_pos_columns

MIN_DIAMETER = 5
SHAPE = (48, 256, 256)  # z, y, x -- thin in z (larger than the unit test)
ASPECT = (1 / 3, 1, 1)  # z one third of x/y
MAX_DIAMETER_BY_RANGE = {'x5': 25, 'x10': 51}  # size-range label -> max x/y diameter
GAP_BY_DENSITY = {'low': 20, 'med': 9, 'high': 3}  # density label -> edge gap (px); smaller = denser
NOISE_STD = 12  # high background noise
POS_COLUMNS = default_pos_columns(3)
MINMASS = 50  # above the noise floor, below any real feature
MATCH_TOLERANCE = 1.5  # max px between a detection and its true position
TARGET = 150  # features per frame (packing target)


def render(shape, centers, sizes, noise_std, seed):
    """Render axis-aligned ellipsoidal Gaussians of given (center, reference Rg).

    Each feature's per-axis radius of gyration is ``size * aspect[i]``.
    """
    image = np.zeros(shape, dtype=np.uint8)
    for center, size in zip(centers, sizes):
        draw_feature(image, center, tuple(size * a for a in ASPECT),
                     max_value=200)
    if noise_std:
        rng = np.random.default_rng(seed)
        image = np.clip(image.astype(float) + rng.normal(0, noise_std, shape),
                        0, 255).astype(np.uint8)
    return image


def pack(max_diameter, gap, seed, target=TARGET, max_attempts=80000):
    """Random non-overlapping ellipsoids spanning [~5, max_diameter] x/y diameters.

    Rejection is done in aspect-normalised coordinates (where the ellipsoids are
    spheres of the reference visible radius), so features never overlap; the
    per-axis placement margin follows ``aspect`` (a thin z axis gets a thin
    margin). Returns (centers, sizes) with size = reference radius of gyration.
    """
    rng = np.random.default_rng(seed)
    max_rg = max_diameter / 3.5
    asp = np.array(ASPECT, dtype=float)
    margin = [int(max_diameter * a) // 2 + 3 for a in ASPECT]
    centers = np.empty((0, len(SHAPE)))
    visible_radii = np.empty(0)
    sizes = []
    for _ in range(max_attempts):
        if len(sizes) >= target:
            break
        radius_of_gyration = rng.uniform(1.4, max_rg)
        visible_radius = 2.0 * radius_of_gyration
        candidate = np.array([rng.uniform(m, s - m)
                              for s, m in zip(SHAPE, margin)])
        normalized = np.sqrt(np.sum(((centers - candidate) / asp) ** 2, axis=1))
        if len(centers) == 0 or np.all(
                normalized >= visible_radii + visible_radius + gap):
            centers = np.vstack([centers, candidate])
            visible_radii = np.append(visible_radii, visible_radius)
            sizes.append(radius_of_gyration)
    return centers, np.array(sizes)


def recall_and_error(features, true_centers):
    """Recall (fraction of true features matched within tolerance), the RMS
    position error over the matched features, and the number matched."""
    if len(features) == 0:
        return 0.0, float('nan'), 0
    distance_to_detection, _ = cKDTree(
        features[POS_COLUMNS].values).query(true_centers)
    matched = distance_to_detection < MATCH_TOLERANCE
    n_matched = int(matched.sum())
    recall = float(np.mean(matched))
    error = (np.sqrt(np.mean(distance_to_detection[matched] ** 2))
             if matched.any() else float('nan'))
    return recall, error, n_matched


def main():
    baseline_seconds = poly_seconds = 0.0
    baseline_correct = poly_correct = total_true = 0
    print("confocal aspect=%s  shape=%s  noise_std=%d" % (
        ASPECT, SHAPE, NOISE_STD))
    print("%-5s %-5s %4s | %-20s %-20s"
          % ("range", "dens", "N", "baseline", "poly"))
    print("-" * 70)
    for range_i, (range_label, max_diameter) in enumerate(
            MAX_DIAMETER_BY_RANGE.items()):
        max_tuple = Polydisperse(MIN_DIAMETER, max_diameter,
                                 aspect=ASPECT).for_ndim(3).max_diameter
        poly = Polydisperse(MIN_DIAMETER, max_diameter, aspect=ASPECT)
        for density_i, (density_label, gap) in enumerate(
                GAP_BY_DENSITY.items()):
            seed = 100 * range_i + 10 * density_i
            true_centers, sizes = pack(max_diameter, gap, seed)
            image = render(SHAPE, true_centers, sizes, NOISE_STD, seed)

            start = time.perf_counter()
            baseline = tp.locate(image, max_tuple, minmass=MINMASS,
                                 characterize=False)
            after_baseline = time.perf_counter()
            found = tp.locate(image, poly, minmass=MINMASS, characterize=False)
            after_poly = time.perf_counter()
            baseline_seconds += after_baseline - start
            poly_seconds += after_poly - after_baseline

            base_recall, base_error, base_n = recall_and_error(
                baseline, true_centers)
            poly_recall, poly_error, poly_n = recall_and_error(
                found, true_centers)
            baseline_correct += base_n
            poly_correct += poly_n
            total_true += len(true_centers)
            # Precision: fraction of a method's detections that are true (matched
            # a distinct feature); the rest are false positives.
            base_prec = base_n / len(baseline) if len(baseline) else float('nan')
            poly_prec = poly_n / len(found) if len(found) else float('nan')
            print("%-5s %-5s %4d | r%.2f e%.3f p%.2f  r%.2f e%.3f p%.2f"
                  % (range_label, density_label, len(true_centers),
                     base_recall, base_error, base_prec,
                     poly_recall, poly_error, poly_prec))

    print("-" * 70)
    print("Total correct (within %.1fpx) of %d true:  baseline=%d  poly=%d  "
          "(poly/baseline=%.2fx)"
          % (MATCH_TOLERANCE, total_true, baseline_correct, poly_correct,
             poly_correct / baseline_correct))
    print("Total detect+refine time:  baseline=%.2fs  poly=%.2fs  "
          "(poly/baseline=%.2fx)"
          % (baseline_seconds, poly_seconds, poly_seconds / baseline_seconds))


if __name__ == '__main__':
    main()
