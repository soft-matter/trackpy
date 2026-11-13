"""Benchmark: polydisperse ``locate`` vs. the naive ``diameter=max`` baseline.

Runs a grid of synthetic images over size range x density x noise and reports,
per cell, each method's recall (``r``), RMS position error over matched features
(``e``) and precision (``p``, the fraction of detections that are true -- the
rest are false positives), plus grid totals and wall-clock time. SPIFF
(``spiff='auto'``) is applied to both methods where a frame has enough features,
and silently skipped otherwise.

Run:  python benchmarks/polydisperse_grid.py
"""
import time

import numpy as np
from scipy.spatial import cKDTree

import trackpy as tp
from trackpy import Polydisperse
from trackpy.artificial import draw_feature
from trackpy.utils import default_pos_columns

MIN_DIAMETER = 5
SHAPE = (500, 500)
MAX_DIAMETER_BY_RANGE = {'x3': 15, 'x5': 25, 'x10': 51}  # size-range label -> max diameter
GAP_BY_DENSITY = {'low': 24, 'med': 10, 'high': 3}  # density label -> edge gap (px); smaller = denser
STD_BY_NOISE = {'low': 1, 'med': 5, 'high': 12}  # noise label -> background std
POS_COLUMNS = default_pos_columns(2)
MINMASS = 500
MATCH_TOLERANCE = 1.5  # max px between a detection and its true position


def render(shape, centers, sizes, noise_std, seed):
    """Render Gaussian features of given (center, radius-of-gyration)."""
    image = np.zeros(shape, dtype=np.uint8)
    for center, size in zip(centers, sizes):
        draw_feature(image, center, size, max_value=200)
    if noise_std:
        rng = np.random.default_rng(seed)
        image = np.clip(image.astype(float) + rng.normal(0, noise_std, shape),
                        0, 255).astype(np.uint8)
    return image


def pack(max_diameter, gap, seed, target=300, max_attempts=15000):
    """Random non-overlapping features spanning [~5, max_diameter] diameters.

    Rejects a candidate whose centre is closer than the sum of the two features'
    visible radii plus ``gap`` to any accepted feature, so a smaller ``gap``
    packs more densely. Returns (centers, sizes) where size = radius of gyration.
    """
    rng = np.random.default_rng(seed)
    max_rg = max_diameter / 3.5
    centers = np.empty((0, 2))
    radii = np.empty(0)
    sizes = []
    low, high = 28, SHAPE[0] - 28
    for _ in range(max_attempts):
        if len(sizes) >= target:
            break
        radius_of_gyration = rng.uniform(1.4, max_rg)
        visible_radius = 2.0 * radius_of_gyration
        candidate = rng.uniform(low, high, 2)
        distances = np.hypot(*(centers - candidate).T)  # to every accepted feature
        if len(centers) == 0 or np.all(distances >= radii + visible_radius + gap):
            centers = np.vstack([centers, candidate])
            radii = np.append(radii, visible_radius)
            sizes.append(radius_of_gyration)
    return centers, np.array(sizes)


def recall_and_error(features, true_centers):
    """Recall (fraction of true features matched within tolerance), the RMS
    position error over the matched features, and the number matched."""
    if len(features) == 0:
        return 0.0, float('nan'), 0
    distance_to_detection, _ = cKDTree(features[POS_COLUMNS].values).query(true_centers)
    matched = distance_to_detection < MATCH_TOLERANCE
    n_matched = int(matched.sum())
    recall = float(np.mean(matched))
    error = (np.sqrt(np.mean(distance_to_detection[matched] ** 2))
             if matched.any() else float('nan'))
    return recall, error, n_matched


def main():
    # Warm up the numba JIT so it doesn't skew the first timed cell.
    warmup_centers, warmup_sizes = pack(25, 10, 0, target=20)
    warmup_image = render(SHAPE, warmup_centers, warmup_sizes, 1, 0)
    tp.locate(warmup_image, 25, minmass=MINMASS, spiff='auto', characterize=False)
    tp.locate(warmup_image, Polydisperse(MIN_DIAMETER, 25), minmass=MINMASS,
              spiff='auto', characterize=False)

    baseline_seconds = poly_seconds = 0.0
    baseline_correct = poly_correct = total_true = 0
    print("%-5s %-5s %-5s %4s | %-20s %-20s"
          % ("range", "dens", "noise", "N", "baseline", "poly"))
    print("-" * 76)
    for range_i, (range_label, max_diameter) in enumerate(MAX_DIAMETER_BY_RANGE.items()):
        for density_i, (density_label, gap) in enumerate(GAP_BY_DENSITY.items()):
            for noise_i, (noise_label, noise_std) in enumerate(STD_BY_NOISE.items()):
                seed = 100 * range_i + 10 * density_i + noise_i
                true_centers, sizes = pack(max_diameter, gap, seed)
                image = render(SHAPE, true_centers, sizes, noise_std, seed)

                start = time.perf_counter()
                baseline = tp.locate(image, max_diameter, minmass=MINMASS,
                                     spiff='auto', characterize=False)
                after_baseline = time.perf_counter()
                poly = tp.locate(image, Polydisperse(MIN_DIAMETER, max_diameter),
                                 minmass=MINMASS, spiff='auto',
                                 characterize=False)
                after_poly = time.perf_counter()
                baseline_seconds += after_baseline - start
                poly_seconds += after_poly - after_baseline

                baseline_recall, baseline_error, baseline_n = recall_and_error(
                    baseline, true_centers)
                poly_recall, poly_error, poly_n = recall_and_error(
                    poly, true_centers)
                baseline_correct += baseline_n
                poly_correct += poly_n
                total_true += len(true_centers)
                # Precision: fraction of a method's detections that are true
                # (matched a distinct feature); the rest are false positives.
                baseline_prec = baseline_n / len(baseline) if len(baseline) else float('nan')
                poly_prec = poly_n / len(poly) if len(poly) else float('nan')
                print("%-5s %-5s %-5s %4d | r%.2f e%.3f p%.2f  r%.2f e%.3f p%.2f"
                      % (range_label, density_label, noise_label, len(true_centers),
                         baseline_recall, baseline_error, baseline_prec,
                         poly_recall, poly_error, poly_prec))

    print("-" * 76)
    print("Total correct (within %.1fpx) of %d true:  baseline=%d  poly=%d  "
          "(poly/baseline=%.2fx)"
          % (MATCH_TOLERANCE, total_true, baseline_correct, poly_correct,
             poly_correct / baseline_correct))
    print("Total grid time:  baseline=%.2fs  poly=%.2fs  (poly/baseline=%.2fx)"
          % (baseline_seconds, poly_seconds, poly_seconds / baseline_seconds))


if __name__ == '__main__':
    main()
