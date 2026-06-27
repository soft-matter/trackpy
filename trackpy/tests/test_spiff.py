import unittest
import warnings

import numpy as np
import pandas as pd

import trackpy as tp
from trackpy.utils import default_pos_columns
from trackpy.artificial import draw_array
from trackpy.spiff import apply_spiff, MIN_FEATURES
from trackpy.tests.common import StrictTestCase, sort_positions


def _subpix_bias(features, columns):
    """Histogram-based measure of sub-pixel bias across position columns."""
    return np.std(
        np.histogram(features[columns].values % 1, bins=10, range=(0, 1))[0]
    )


class TestSpiff(StrictTestCase):
    def _test_spiff(self, ndim):
        # Draw an image with 200 features and some noise
        expected, image = draw_array(200, 2, noise_level=1, ndim=ndim)

        columns = default_pos_columns(ndim)

        # Locate the features and calculate the deviation
        features = tp.locate(image, diameter=5)
        _, actual = sort_positions(features[columns].values, expected)
        deviation = np.sqrt(np.mean(np.sum((actual - expected) ** 2, 1)))

        # Apply SPIFF correction and calculate the deviation
        corrected_features = apply_spiff(features)
        _, corrected = sort_positions(corrected_features[columns].values, expected)
        deviation_corrected = np.sqrt(np.mean(np.sum((corrected - expected) ** 2, 1)))

        # Verify that the SPIFF correction improves accuracy
        assert (deviation_corrected < deviation)

        # Verify that the SPIFF corrected features have less subpixel bias
        hist_dev = _subpix_bias(features, columns)
        corrected_hist_dev = _subpix_bias(corrected_features, columns)
        assert (corrected_hist_dev < hist_dev)

    def test_spiff_2d(self):
        self._test_spiff(ndim=2)

    def test_spiff_3d(self):
        self._test_spiff(ndim=3)


class TestSpiffOption(StrictTestCase):
    """Tests for the ``spiff`` option of ``locate`` and ``batch``."""

    def _make_image(self, n=200, ndim=2):
        return draw_array(n, 2, noise_level=1, ndim=ndim)

    def test_locate_spiff_false_default(self):
        # By default, locate must not modify features via SPIFF.
        expected, image = self._make_image()
        baseline = tp.locate(image, diameter=5)
        corrected = tp.locate(image, diameter=5, spiff=False)
        pd.testing.assert_frame_equal(baseline, corrected)

    def test_locate_spiff_true_reduces_bias(self):
        expected, image = self._make_image()
        columns = default_pos_columns(2)
        baseline = tp.locate(image, diameter=5)
        corrected = tp.locate(image, diameter=5, spiff=True)
        # Same number of features, but corrected positions should have
        # less sub-pixel bias.
        assert len(baseline) == len(corrected)
        assert _subpix_bias(corrected, columns) < _subpix_bias(baseline, columns)
        # And the values should actually be different.
        assert not np.allclose(baseline[columns].values,
                               corrected[columns].values)

    def test_locate_spiff_true_warns_on_few_features(self):
        # A small image yields very few features; spiff=True should warn.
        expected, image = self._make_image(n=4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = tp.locate(image, diameter=5, spiff=True)
        messages = [str(w.message) for w in caught]
        assert any("SPIFF" in m for m in messages), messages
        assert len(result) < MIN_FEATURES

    def test_locate_spiff_auto_silent_on_few_features(self):
        expected, image = self._make_image(n=4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tp.locate(image, diameter=5, spiff='auto')
        messages = [str(w.message) for w in caught]
        assert not any("SPIFF" in m for m in messages), messages

    def test_batch_spiff_pools_across_frames(self):
        # Build a small "video" of identical frames so SPIFF gets enough
        # features when pooled across all of them.
        expected, image = self._make_image(n=50)
        frames = [image, image, image, image]
        columns = default_pos_columns(2)
        baseline = tp.batch(frames, diameter=5)
        corrected = tp.batch(frames, diameter=5, spiff=True)
        assert len(baseline) == len(corrected)
        assert _subpix_bias(corrected, columns) < _subpix_bias(baseline, columns)

    def test_batch_spiff_default_is_off(self):
        expected, image = self._make_image(n=50)
        frames = [image, image]
        baseline = tp.batch(frames, diameter=5)
        explicit = tp.batch(frames, diameter=5, spiff=False)
        pd.testing.assert_frame_equal(baseline, explicit)

    def test_batch_spiff_with_output_applies_per_frame(self):
        # When `output` is specified, features cannot be pooled across
        # frames, so SPIFF should fall back to per-frame application via
        # the underlying `locate` call.
        expected, image = self._make_image(n=50)
        frames = [image, image]
        columns = default_pos_columns(2)

        class Collector:
            def __init__(self):
                self.frames = []

            def put(self, features):
                self.frames.append(features)

        baseline = Collector()
        tp.batch(frames, diameter=5, output=baseline)
        corrected = Collector()
        tp.batch(frames, diameter=5, spiff=True, output=corrected)

        baseline_all = pd.concat(baseline.frames, ignore_index=True)
        corrected_all = pd.concat(corrected.frames, ignore_index=True)
        assert len(baseline_all) == len(corrected_all)
        assert (_subpix_bias(corrected_all, columns)
                < _subpix_bias(baseline_all, columns))


if __name__ == '__main__':
    unittest.main()
