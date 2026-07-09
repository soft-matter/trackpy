"""Benchmark: polydisperse ``locate`` vs. the naive ``diameter=max`` baseline.

Runs a grid of synthetic images over size range x density x noise and reports,
per cell, the recall and position error of each method, plus the total
wall-clock time for the whole grid. SPIFF (``spiff='auto'``) is applied to both
methods where a frame has enough features, and silently skipped otherwise.

Run:  python benchmarks/polydisperse_grid.py
"""
import time

import numpy as np
from scipy.spatial import cKDTree

import trackpy as tp
from trackpy import Polydisperse
from trackpy.artificial import draw_feature
from trackpy.utils import default_pos_columns

MIN_D = 5
SHAPE = (500, 500)
RANGES = {'x3': 15, 'x5': 25, 'x10': 51}     # max_diameter (min_diameter = 5)
DENS_GAP = {'low': 24, 'med': 10, 'high': 3}   # edge gap in px (smaller = denser)
NOISE = {'low': 1, 'med': 5, 'high': 12}       # background noise std
COLS = default_pos_columns(2)
MINMASS = 80
TOL = 1.5                                      # match distance, px


def draw(shape, positions, sizes, noise, seed):
    """Render Gaussian features of given (position, radius-of-gyration)."""
    image = np.zeros(shape, dtype=np.uint8)
    for position, size in zip(positions, sizes):
        draw_feature(image, position, size, max_value=200)
    if noise:
        rng = np.random.default_rng(seed)
        image = np.clip(image.astype(float) + rng.normal(0, noise, shape),
                        0, 255).astype(np.uint8)
    return image


def pack(max_diameter, gap, seed, target=300, tries=15000):
    """Random non-overlapping features spanning [~5, max_diameter] diameters."""
    rng = np.random.default_rng(seed)
    rg_hi = max_diameter / 3.5
    pos, rad, sizes = np.empty((0, 2)), np.empty(0), []
    lo, hi = 28, SHAPE[0] - 28
    for _ in range(tries):
        if len(sizes) >= target:
            break
        rg = rng.uniform(1.4, rg_hi)
        r = 2.0 * rg                            # visible feature radius
        c = rng.uniform(lo, hi, 2)
        if len(pos) == 0 or np.all(np.hypot(*(pos - c).T) >= rad + r + gap):
            pos = np.vstack([pos, c])
            rad = np.append(rad, r)
            sizes.append(rg)
    return pos, np.array(sizes)


def stats(features, positions):
    """Recall (fraction of truth matched within TOL) and matched position RMS."""
    if len(features) == 0:
        return 0.0, float('nan')
    d, _ = cKDTree(features[COLS].values).query(positions)
    hit = d < TOL
    rms = np.sqrt(np.mean(d[hit] ** 2)) if hit.any() else float('nan')
    return float(np.mean(hit)), rms


def main():
    # Warm up the numba JIT so it doesn't skew the first timed cell.
    warm_pos, warm_sizes = pack(25, 10, 0, target=20)
    warm = draw(SHAPE, warm_pos, warm_sizes, 1, 0)
    tp.locate(warm, 25, minmass=MINMASS, spiff='auto')
    tp.locate(warm, Polydisperse(MIN_D, 25), minmass=MINMASS, spiff='auto')

    base_time = poly_time = 0.0
    print("%-5s %-5s %-5s %4s | %-13s %-13s"
          % ("range", "dens", "noise", "N", "baseline", "poly"))
    print("-" * 62)
    for ri, (range_name, max_d) in enumerate(RANGES.items()):
        for di, dens in enumerate(DENS_GAP):
            for ni, noise in enumerate(NOISE):
                seed = 100 * ri + 10 * di + ni
                pos, sizes = pack(max_d, DENS_GAP[dens], seed)
                image = draw(SHAPE, pos, sizes, NOISE[noise], seed)

                t0 = time.perf_counter()
                base = tp.locate(image, max_d, minmass=MINMASS, spiff='auto')
                t1 = time.perf_counter()
                poly = tp.locate(image, Polydisperse(MIN_D, max_d),
                                 minmass=MINMASS, spiff='auto')
                t2 = time.perf_counter()
                base_time += t1 - t0
                poly_time += t2 - t1

                rb, eb = stats(base, pos)
                rp, ep = stats(poly, pos)
                print("%-5s %-5s %-5s %4d | r%.2f e%.3f  r%.2f e%.3f"
                      % (range_name, dens, noise, len(pos), rb, eb, rp, ep))

    print("-" * 62)
    print("Total grid time:  baseline=%.2fs  poly=%.2fs  (poly/baseline=%.2fx)"
          % (base_time, poly_time, poly_time / base_time))


if __name__ == '__main__':
    main()
