import numpy as np

from numpy.testing import assert_equal

from trackpy.masks import slice_image, mask_image
from trackpy.tests.common import StrictTestCase


class TestSlicing(StrictTestCase):
    def test_slicing_2D(self):
        im = np.empty((9, 9))

        # center
        for radius in range(1, 5):
            sli, origin = slice_image([4, 4], im, radius)
            assert_equal(sli.shape, (radius*2+1,) * 2)
            assert_equal(origin, (4 - radius,) * 2)

        # edge
        for radius in range(1, 5):
            sli, origin = slice_image([0, 4], im, radius)
            assert_equal(sli.shape, (radius + 1, radius*2+1))

        # edge
        for radius in range(1, 5):
            sli, origin = slice_image([4, 0], im, radius)
            assert_equal(sli.shape, (radius*2+1, radius + 1))

        # corner
        for radius in range(1, 5):
            sli, origin = slice_image([0, 0], im, radius)
            assert_equal(sli.shape, (radius+1, radius + 1))

        # outside of image
        for radius in range(2, 5):
            sli, origin = slice_image([-1, 4], im, radius)
            assert_equal(sli.shape, (radius, radius*2+1))

        # outside of image
        for radius in range(2, 5):
            sli, origin = slice_image([-1, -1], im, radius)
            assert_equal(sli.shape, (radius, radius))

        # no slice
        for radius in range(2, 5):
            sli, origin = slice_image([-10, 20], im, radius)
            assert_equal(sli.shape, (0, 0))


    def test_slicing_3D(self):
        im = np.empty((9, 9, 9))

        # center
        for radius in range(1, 5):
            sli, origin = slice_image([4, 4, 4], im, radius)
            assert_equal(sli.shape, (radius*2+1,) * 3)
            assert_equal(origin, (4 - radius,) * 3)

        # face
        for radius in range(1, 5):
            sli, origin = slice_image([0, 4, 4], im, radius)
            assert_equal(sli.shape, (radius + 1, radius*2+1, radius*2+1))

        # edge
        for radius in range(1, 5):
            sli, origin = slice_image([4, 0, 0], im, radius)
            assert_equal(sli.shape, (radius*2+1, radius + 1, radius + 1))

        # corner
        for radius in range(1, 5):
            sli, origin = slice_image([0, 0, 0], im, radius)
            assert_equal(sli.shape, (radius+1, radius + 1, radius + 1))

        # outside of image
        for radius in range(2, 5):
            sli, origin = slice_image([-1, 4, 4], im, radius)
            assert_equal(sli.shape, (radius, radius*2+1, radius*2+1))

        # outside of image
        for radius in range(2, 5):
            sli, origin = slice_image([-1, -1, 4], im, radius)
            assert_equal(sli.shape, (radius, radius, radius*2+1))

        # no slice
        for radius in range(2, 5):
            sli, origin = slice_image([-10, 20, 30], im, radius)
            assert_equal(sli.shape, (0, 0, 0))

    def test_slicing_2D_multiple(self):
        im = np.empty((9, 9))
        radius = 2

        sli, origin = slice_image([[4, 4], [4, 4]], im, radius)
        assert_equal(sli.shape, (5, 5))
        assert_equal(origin, (2, 2))

        sli, origin = slice_image([[4, 2], [4, 6]], im, radius)
        assert_equal(sli.shape, (5, 9))
        assert_equal(origin, (2, 0))

        sli, origin = slice_image([[2, 4], [6, 4]], im, radius)
        assert_equal(sli.shape, (9, 5))
        assert_equal(origin, (0, 2))

        sli, origin = slice_image([[2, 4], [6, 4], [-10, 20]], im, radius)
        assert_equal(sli.shape, (9, 5))
        assert_equal(origin, (0, 2))

    def test_slicing_3D_multiple(self):
        im = np.empty((9, 9, 9))
        radius = 2

        sli, origin = slice_image([[4, 4, 4], [4, 4, 4]], im, radius)
        assert_equal(sli.shape, (5, 5, 5))
        assert_equal(origin, (2, 2, 2))

        sli, origin = slice_image([[4, 2, 4], [4, 6, 4]], im, radius)
        assert_equal(sli.shape, (5, 9, 5))
        assert_equal(origin, (2, 0, 2))

        sli, origin = slice_image([[4, 2, 6], [4, 6, 2]], im, radius)
        assert_equal(sli.shape, (5, 9, 9))
        assert_equal(origin, (2, 0, 0))

        sli, origin = slice_image([[4, 2, 4], [4, 6, 4], [-10, 4, 4]], im, radius)
        assert_equal(sli.shape, (5, 9, 5))
        assert_equal(origin, (2, 0, 2))


class TestMasking(StrictTestCase):
    def test_masking_single_2D(self):
        im = np.ones((9, 9))
        radius = 1  # N pix is 5

        sli = mask_image([4, 4], im, radius)
        assert_equal(sli.sum(), 5)
        assert_equal(sli.shape, im.shape)

        sli = mask_image([0, 4], im, radius)
        assert_equal(sli.sum(), 4)

        sli = mask_image([4, 0], im, radius)
        assert_equal(sli.sum(), 4)

        sli = mask_image([0, 0], im, radius)
        assert_equal(sli.sum(), 3)

        sli = mask_image([-1, 4], im, radius)
        assert_equal(sli.sum(), 1)

        sli = mask_image([-1, -1], im, radius)
        assert_equal(sli.sum(), 0)


    def test_masking_multiple_2D(self):
        im = np.ones((9, 9))
        radius = 1  # N pix is 5

        sli = mask_image([[4, 2], [4, 6]], im, radius)
        assert_equal(sli.sum(), 10)

        sli = mask_image([[4, 4], [4, 4]], im, radius)
        assert_equal(sli.sum(), 5)

        sli = mask_image([[0, 4], [4, 4]], im, radius)
        assert_equal(sli.sum(), 9)

        sli = mask_image([[-1, 4], [4, 4]], im, radius)
        assert_equal(sli.sum(), 6)

        sli = mask_image([[-20, 40], [4, 4]], im, radius)
        assert_equal(sli.sum(), 5)

    def test_masking_single_3D(self):
        im = np.ones((9, 9, 9))
        radius = 1  # N pix is 7

        sli = mask_image([4, 4, 4], im, radius)
        assert_equal(sli.sum(), 7)
        assert_equal(sli.shape, im.shape)

        sli = mask_image([0, 4, 4], im, radius)
        assert_equal(sli.sum(), 6)

        sli = mask_image([4, 0, 0], im, radius)
        assert_equal(sli.sum(), 5)

        sli = mask_image([0, 0, 0], im, radius)
        assert_equal(sli.sum(), 4)

        sli = mask_image([-1, 4, 4], im, radius)
        assert_equal(sli.sum(), 1)

        sli = mask_image([-1, -1, -1], im, radius)
        assert_equal(sli.sum(), 0)

    def test_masking_multiple_3D(self):
        im = np.ones((9, 9, 9))
        radius = 1  # N pix is 7

        sli = mask_image([[4, 4, 4], [4, 4, 4]], im, radius)
        assert_equal(sli.sum(), 7)
        assert_equal(sli.shape, im.shape)

        sli = mask_image([[4, 4, 6], [4, 4, 2]], im, radius)
        assert_equal(sli.sum(), 14)

        sli = mask_image([[4, 4, 0], [4, 4, 4]], im, radius)
        assert_equal(sli.sum(), 13)


if __name__ == '__main__':
    import unittest
    unittest.main()
