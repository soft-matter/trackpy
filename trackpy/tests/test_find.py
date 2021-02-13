import numpy as np

import trackpy as tp
from trackpy.artificial import draw_feature
from trackpy.tests.common import assert_coordinates_close, StrictTestCase
from trackpy.find import grey_dilation, grey_dilation_legacy

try:
    from pandas._testing import assert_produces_warning
except ImportError:
    from pandas.util.testing import assert_produces_warning


class TestFindGreyDilation(StrictTestCase):
    def test_separation_fast(self):
        separation = 20
        for angle in np.arange(0, 360, 15):
            im = np.zeros((128, 128), dtype=np.uint8)
            pos = [[64, 64], [64 + separation * np.sin(angle/180*np.pi),
                              64 + separation * np.cos(angle/180*np.pi)]]

            # setup features: features with equal signal will always be
            # detected by a grey dilation, so make them unequal
            draw_feature(im, pos[0], 3, 240)
            draw_feature(im, pos[1], 3, 250)

            # find both of them
            f = grey_dilation(im, separation - 1, precise=False)
            assert_coordinates_close(f, pos, atol=1)

            # find only the brightest
            if angle in [45, 135, 225, 315]:
                # for unprecise, a too small square kernel is used, which is
                # perfect for 45-degree angles
                f = grey_dilation(im, separation + 1, precise=False)
                assert_coordinates_close(f, pos[1:], atol=1)
            else:
                # but too small by a factor of sqrt(ndim) for 90-degree angles
                f = grey_dilation(im, separation*np.sqrt(2) + 1, precise=False)
                assert_coordinates_close(f, pos[1:], atol=1)


    def test_separation(self):
        separation = 20
        for angle in np.arange(0, 360, 15):
            im = np.zeros((128, 128), dtype=np.uint8)
            pos = [[64, 64], [64 + separation * np.sin(angle/180*np.pi),
                              64 + separation * np.cos(angle/180*np.pi)]]

            # setup features: features with equal signal will always be
            # detected by a grey dilation, so make them unequal
            draw_feature(im, pos[0], 3, 240)
            draw_feature(im, pos[1], 3, 250)

            # find both of them
            f = grey_dilation(im, separation - 1)
            assert_coordinates_close(f, pos, atol=1)

            # find only the brightest
            f = grey_dilation(im, separation + 1)
            assert_coordinates_close(f, pos[1:], atol=1)

    def test_separation_anisotropic(self):
        separation = (10, 20)
        for angle in np.arange(0, 360, 15):
            im = np.zeros((128, 128), dtype=np.uint8)
            pos = [[64, 64], [64 + separation[0] * np.sin(angle/180*np.pi),
                              64 + separation[1] * np.cos(angle/180*np.pi)]]

            # setup features: features with equal signal will always be
            # detected by a grey dilation, so make them unequal
            draw_feature(im, pos[0], 3, 240)
            draw_feature(im, pos[1], 3, 250)

            # find both of them
            f = grey_dilation(im, (9, 19))
            assert_coordinates_close(f, pos, atol=1)

            # find only the brightest
            f = grey_dilation(im, (11, 21))
            assert_coordinates_close(f, pos[1:], atol=1)

    def test_float_image(self):
        separation = 20
        angle = 45
        im = np.zeros((128, 128), dtype=np.float64)
        pos = [[64, 64], [64 + separation * np.sin(angle/180*np.pi),
                          64 + separation * np.cos(angle/180*np.pi)]]

        # setup features: features with equal signal will always be
        # detected by a grey dilation, so make them unequal
        draw_feature(im, pos[0], 3, 240)
        draw_feature(im, pos[1], 3, 250)

        # find both of them
        f = grey_dilation(im, separation - 1, precise=False)
        assert_coordinates_close(f, pos, atol=1)



class TestFindGreyDilationLegacy(StrictTestCase):
    def test_separation(self):
        separation = 20
        for angle in np.arange(0, 360, 15):
            im = np.zeros((128, 128), dtype=np.uint8)
            pos = [[64, 64], [64 + separation * np.sin(angle/180*np.pi),
                              64 + separation * np.cos(angle/180*np.pi)]]

            # setup features: features with equal signal will always be
            # detected by grey_dilation_legacy, so make them unequal
            draw_feature(im, pos[0], 3, 240)
            draw_feature(im, pos[1], 3, 250)

            # find both of them
            f = grey_dilation_legacy(im, separation - 1)
            assert_coordinates_close(f, pos, atol=1)

            # find only the brightest
            f = grey_dilation_legacy(im, separation + 1)
            assert_coordinates_close(f, pos[1:], atol=1)


if __name__ == '__main__':
    import unittest
    unittest.main()
