from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import unittest
import numpy as np

from numpy.testing import assert_almost_equal, assert_allclose, assert_equal
import pandas as pd

from scipy.spatial import cKDTree

import trackpy as tp
from trackpy.masks import slice_image, mask_image
from trackpy.artificial import SimulatedImage


def sort_positions(actual, expected):
    assert_equal(len(actual), len(expected))
    tree = cKDTree(actual)
    devs, argsort = tree.query([expected])
    return devs, actual[argsort][0]


def assert_coordinates_close(actual, expected, atol):
    _, sorted_actual = sort_positions(actual, expected)
    assert_allclose(sorted_actual, expected, atol)


def dummy_cluster(N, center, separation, ndim=2):
    devs = (np.random.random((N, ndim)) - 0.5) * separation / np.sqrt(ndim)
    return np.array(center)[np.newaxis, :] + devs


def dummy_clusters(N, max_size, separation, ndim=2):
    center = [separation] * ndim
    sizes = np.random.randint(1, max_size, N)
    displ = (np.random.random((N, ndim)) + 2) * separation
    res = []
    for i, size in enumerate(sizes):
        center += displ[i]
        res.append(dummy_cluster(size, center, separation, ndim))
    return res


def pos_to_df(pos):
    pos_a = np.concatenate(pos)
    ndim = pos_a.shape[1]
    pos_columns = ['z', 'y', 'x'][-ndim:]
    return pd.DataFrame(pos_a, columns=pos_columns)


class TestSlicing(unittest.TestCase):
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


class TestMasking(unittest.TestCase):
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


class TestFindClusters(unittest.TestCase):
    def setUp(self):
        self.N = 10

    def test_single_cluster_2D(self):
        separation = np.random.random(self.N) * 10
        for sep in separation:
            pos = dummy_clusters(1, 10, sep)
            df = pos_to_df(pos)
            df = tp.find_clusters(df, sep)
            assert_equal(df['cluster_size'].values, len(pos[0]))

    def test_multiple_clusters_2D(self):
        numbers = np.random.randint(1, 10, self.N)
        for number in numbers:
            pos = dummy_clusters(number, 10, 1)
            df = pos_to_df(pos)
            df = tp.find_clusters(df, 1)
            assert_equal(df['cluster'].nunique(), number)

    def test_single_cluster_3D(self):
        separation = np.random.random(self.N) * 10
        for sep in separation:
            pos = dummy_clusters(1, 10, sep, 3)
            df = pos_to_df(pos)
            df = tp.find_clusters(df, sep)
            assert_equal(df['cluster_size'].values, len(pos[0]))

    def test_multiple_clusters_3D(self):
        numbers = np.random.randint(1, 10, self.N)
        for number in numbers:
            pos = dummy_clusters(number, 10, 1, 3)
            df = pos_to_df(pos)
            df = tp.find_clusters(df, 1)
            assert_equal(df['cluster'].nunique(), number)

    def test_line_cluster(self):
        separation = np.random.random(self.N) * 10
        angle = np.random.random(self.N) * 2 * np.pi
        ds = np.array([np.cos(angle), np.sin(angle)]).T
        for vec, sep in zip(ds, separation):
            pos = np.arange(10)[:, np.newaxis] * vec[np.newaxis, :] * sep
            df = pos_to_df([pos])
            df = tp.find_clusters(df, sep*1.1)
            assert_equal(df['cluster_size'].values, 10)
            df = tp.find_clusters(df, sep*0.9)
            assert_equal(df['cluster_size'].values, 1)

            df = pos_to_df([pos[::-1]])
            df = tp.find_clusters(df, sep*1.1)
            assert_equal(df['cluster_size'].values, 10)
            df = tp.find_clusters(df, sep*0.9)
            assert_equal(df['cluster_size'].values, 1)

            ind = np.arange(10)
            np.random.shuffle(ind)
            pos = ind[:, np.newaxis] * vec[np.newaxis, :] * sep
            df = pos_to_df([pos])
            df = tp.find_clusters(df, sep*1.1)
            assert_equal(df['cluster_size'].values, 10)
            df = tp.find_clusters(df, sep*0.9)
            assert_equal(df['cluster_size'].values, 1)

    def test_line_cluster_3D(self):
        separation = np.random.random(self.N) * 10
        phi = np.random.random(self.N) * 2 * np.pi
        theta = np.random.random(self.N) * np.pi
        ds = np.array([np.cos(theta),
                       np.cos(phi)*np.sin(theta),
                       np.sin(phi)*np.sin(theta)]).T
        for vec, sep in zip(ds, separation):
            pos = np.arange(10)[:, np.newaxis] * vec[np.newaxis, :] * sep
            df = pos_to_df([pos])
            df = tp.find_clusters(df, sep*1.1)
            assert_equal(df['cluster_size'].values, 10)
            df = tp.find_clusters(df, sep*0.9)
            assert_equal(df['cluster_size'].values, 1)

            df = pos_to_df([pos[::-1]])
            df = tp.find_clusters(df, sep*1.1)
            assert_equal(df['cluster_size'].values, 10)
            df = tp.find_clusters(df, sep*0.9)
            assert_equal(df['cluster_size'].values, 1)

            ind = np.arange(10)
            np.random.shuffle(ind)
            pos = ind[:, np.newaxis] * vec[np.newaxis, :] * sep
            df = pos_to_df([pos])
            df = tp.find_clusters(df, sep*1.1)
            assert_equal(df['cluster_size'].values, 10)
            df = tp.find_clusters(df, sep*0.9)
            assert_equal(df['cluster_size'].values, 1)



def refine_single(im, diameter, separation, pos=None,
                  var_size=False, var_signal=False,
                  p0_pos=None, p0_size=None, p0_signal=None,
                  pos_atol=0.1, signal_rtol=0.05, size_rtol=0.05,
                  noise_size=None, threshold=None):
    if pos is None:
        pos = im.center
    im.clear()
    im.draw_feature(pos)
    f0 = im.f()
    if p0_pos is not None:
        f0[im.pos_columns] = p0_pos
    if p0_size is not None:
        if im.isotropic:
            try:
                f0[im.size_columns[0]] = p0_size[0]
            except TypeError:
                f0[im.size_columns[0]] = p0_size
        else:
            f0[im.size_columns] = p0_size

    if p0_signal is not None:
        f0['signal'] = p0_signal

    actual = tp.refine_gaussian(f0, im(), diameter, separation, var_size,
                                var_signal, im.pos_columns, 'frame', False,
                                noise_size, threshold)
    assert actual['gaussian'].iloc[0]
    if var_signal:
        assert_allclose(actual['signal'].iloc[0], im.signal, rtol=signal_rtol)
    if var_size:
        if im.isotropic:
            assert_allclose(actual[im.size_columns[0]].iloc[0], im.size[0],
                            rtol=size_rtol)
        else:
            assert_allclose(actual[im.size_columns].iloc[0], im.size,
                            rtol=size_rtol)

    actual_pos = actual[im.pos_columns].iloc[0].values
    assert_allclose(actual_pos, np.array(pos), atol=pos_atol)

    return actual_pos - pos


class TestSingle(object):
    def setUp(self):
        self.separation = 100
        self.signal = 200
        if not hasattr(self, 'size'):
            if hasattr(self.diameter, '__iter__'):
                self.size = tuple([d / 8 for d in self.diameter])
            else:
                self.size = self.diameter / 8
        self.im = SimulatedImage(self.shape, self.size, dtype=np.uint8,
                                 signal=self.signal,
                                 feat_func=SimulatedImage.feat_gauss)
        self.N = 10

    def test_perfect_gaussian(self):
        pos_err = self.pos_err
        positions = self.im.center + \
                    np.random.random((self.N, self.im.ndim)) * pos_err * 2 - pos_err
        for i, pos in enumerate(positions):
            refine_single(self.im, self.diameter, self.separation, pos,
                          self.var_size, self.var_signal, self.im.center)

    def test_perfect_gaussian_noisy(self):
        pos_err = self.pos_err
        self.im.noise = 20
        positions = self.im.center + \
                    np.random.random((self.N, self.im.ndim)) * pos_err * 2 - pos_err
        for i, pos in enumerate(positions):
            refine_single(self.im, self.diameter, self.separation, pos,
                          self.var_size, self.var_signal, self.im.center,
                          noise_size=1, threshold=20, pos_atol=0.1,
                          signal_rtol=1, size_rtol=1)

    def test_disc_like(self):
        self.im.feat_func = SimulatedImage.feat_hat
        pos_err = self.pos_err
        positions = self.im.center + \
                    np.random.random((self.N, self.im.ndim)) * pos_err * 2 - pos_err
        for i, pos in enumerate(positions):
            for disc_size in [0.1, 0.2]:
                self.im.feat_kwargs = dict(disc_size=disc_size)
                refine_single(self.im, self.diameter, self.separation, pos,
                              self.var_size, self.var_signal, self.im.center,
                              pos_atol=0.2, signal_rtol=10, size_rtol=10)

    def test_changing_size(self):
        if not hasattr(self, 'size_err_factor'):
            raise unittest.SkipTest()
        size_min = self.size_err_factor[0]
        size_range = self.size_err_factor[1] - size_min
        size_err = np.random.random(self.N) * size_range + size_min
        sizes = [[s * err for s in self.im.size] for err in size_err]

        for i, size in enumerate(sizes):
            # don't test signal, it will be different because of wrong size
            refine_single(self.im, self.diameter, self.separation, None,
                          self.var_size, self.var_signal, self.p0_pos, size,
                          signal_rtol=10)

    def test_changing_mass(self):
        if not hasattr(self, 'signal_err_factor'):
            raise unittest.SkipTest()
        signal_min = self.signal_err_factor[0]
        signal_range = self.signal_err_factor[1] - signal_min
        signal_err = np.random.random(self.N) * signal_range + signal_min
        signals = self.signal * signal_err
        for i, signal in enumerate(signals):
            refine_single(self.im, self.diameter, self.separation, None,
                          self.var_size, self.var_signal, self.p0_pos, None,
                          signal, size_rtol=1)

    def test_dimer(self):
        self.im.signal = 100
        if self.im.ndim == 2:
            angles = np.random.random(self.N) * 180
        elif self.im.ndim == 3:
            angles = np.random.random((self.N, 2)) * 180

        for angle in angles:
            for hard_radius in [0.8, 1., 1.2]:
                self.im.clear()
                self.im.draw_dimer(self.im.center, angle, hard_radius)

                f0 = self.im.f(noise=1)
                f0['cluster'] = 0
                actual = tp.refine_gaussian(f0, self.im(), self.diameter,
                                            self.separation, self.var_size,
                                            self.var_signal,
                                            self.im.pos_columns,
                                            'frame', False, None)
                assert np.all(actual['gaussian'])
                assert_coordinates_close(actual[self.im.pos_columns].values,
                                         self.im.coords, atol=0.1)
                if self.var_signal:
                    assert_allclose(actual['signal'], self.im.signal, rtol=0.05)
                if self.var_size:
                    if self.im.isotropic:
                        assert_allclose(actual[self.im.size_columns[0]],
                                        self.im.size[0], rtol=0.05)
                    else:
                        assert_allclose(actual[self.im.size_columns],
                                        [self.im.size] * 2, rtol=0.05)

    def tearDown(self):
        pass


class TestFit_gauss2D(TestSingle, unittest.TestCase):
    shape = (128, 128)
    fit_function = 'gauss'
    var_size = False
    var_signal = False
    diameter = 21
    pos_err = 7
    p0_pos = [64.3, 64.9]
    size_err_factor = [0.5, 2.]
    signal_err_factor = [0.1, 10.]


class TestFit_gauss2D_a(TestSingle, unittest.TestCase):
    shape = (128, 64)
    fit_function = 'gauss'
    var_size = False
    var_signal = False
    diameter = (21, 15)
    pos_err = [7, 4]
    p0_pos = [64.3, 32.9]
    size_err_factor = [0.8, 2.]
    signal_err_factor = [0.1, 10.]


class TestFit_gauss2D_signal(TestSingle, unittest.TestCase):
    # in this test, the magnitude of signal is tested (5% tol)
    shape = (128, 128)
    fit_function = 'gauss'
    var_size = False
    var_signal = True
    diameter = 21
    pos_err = 9
    p0_pos = [64.3, 64.9]
    size_err_factor = [0.5, 2.]  # signal value is untested
    signal_err_factor = [0.1, 10.]


class TestFit_gauss2D_a_signal(TestSingle, unittest.TestCase):
    shape = (128, 64)
    fit_function = 'gauss'
    var_size = False
    var_signal = True
    diameter = (21, 15)
    pos_err = [9, 4]
    p0_pos = [64.3, 32.9]
    size_err_factor = [0.8, 2.]  # signal value is untested
    signal_err_factor = [0.1, 10.]


class TestFit_gauss2D_size(TestSingle, unittest.TestCase):
    shape = (128, 128)
    fit_function = 'gauss'
    var_size = True
    var_signal = False
    diameter = 21
    pos_err = 3  # size diverges easily when pos is not guessed accurately
    p0_pos = [64.3, 64.9]
    size_err_factor = [0.5, 2.]
   # signal_err_factor = [0.1, 10.]


class TestFit_gauss2D_a_size(TestSingle, unittest.TestCase):
    shape = (128, 64)
    fit_function = 'gauss'
    var_size = True
    var_signal = False
    diameter = (21, 15)
    pos_err = [3, 1]  # size diverges easily when pos is not guessed accurately
    p0_pos = [64.3, 32.9]
    size_err_factor = [0.5, 2.]
   # signal_err_factor = [0.1, 10.]


class TestFit_gauss2D_signal_size(TestSingle, unittest.TestCase):
    shape = (128, 128)
    fit_function = 'gauss'
    var_size = True
    var_signal = True
    diameter = 21
    pos_err = 2  # size diverges easily when pos is not guessed accurately
    p0_pos = [64.3, 64.9]
    size_err_factor = [0.5, 2.]
    signal_err_factor = [0.1, 10.]


class TestFit_gauss2D_a_signal_size(TestSingle, unittest.TestCase):
    shape = (128, 64)
    fit_function = 'gauss'
    var_size = True
    var_signal = True
    diameter = (21, 15)
    pos_err = [3, 1]  # size diverges easily when pos is not guessed accurately
    p0_pos = [64.3, 32.9]
    size_err_factor = [0.5, 2.]
    signal_err_factor = [0.1, 10.]


class TestFit_gauss3D(TestSingle, unittest.TestCase):
    shape = (64, 64, 64)
    fit_function = 'gauss'
    var_size = False
    var_signal = False
    diameter = 21
    pos_err = 5
    p0_pos = [32.3, 32.9, 31.7]
    size_err_factor = [0.5, 2.]
    signal_err_factor = [0.1, 10.]


class TestFit_gauss3D_a(TestSingle, unittest.TestCase):
    shape = (64, 64, 32)
    fit_function = 'gauss'
    var_size = False
    var_signal = False
    diameter = (21, 21, 15)
    pos_err = [5, 5, 2]
    p0_pos = [32.3, 32.9, 16.7]
    size_err_factor = [0.5, 2.]
    signal_err_factor = [0.2, 5.]


class TestFit_gauss3D_signal(TestSingle, unittest.TestCase):
    shape = (64, 64, 64)
    fit_function = 'gauss'
    var_size = False
    var_signal = True
    diameter = 21
    pos_err = 7
    p0_pos = [32.3, 32.9, 31.7]
    size_err_factor = [0.5, 2.] # signal value is untested
    signal_err_factor = [0.1, 10.]


class TestFit_gauss3D_a_signal(TestSingle, unittest.TestCase):
    shape = (64, 64, 32)
    fit_function = 'gauss'
    var_size = False
    var_signal = True
    diameter = (21, 21, 15)
    pos_err = [7, 7, 3]
    p0_pos = [32.3, 32.9, 16.7]
    size_err_factor = [0.8, 1.5] # signal value is untested
    signal_err_factor = [0.1, 10.]


class TestFit_gauss3D_size(TestSingle, unittest.TestCase):
    shape = (64, 64, 64)
    fit_function = 'gauss'
    var_size = True
    var_signal = False
    diameter = 21
    pos_err = 2
    p0_pos = [32.3, 32.9, 31.7]
    size_err_factor = [0.2, 5.]
   # signal_err_factor = [0.1, 10.]


class TestFit_gauss3D_a_size(TestSingle, unittest.TestCase):
    shape = (64, 64, 32)
    fit_function = 'gauss'
    var_size = True
    var_signal = False
    diameter = (21, 21, 15)
    pos_err = [2, 2, 1]
    p0_pos = [32.3, 32.9, 16.7]
    size_err_factor = [0.5, 2]
   # signal_err_factor = [0.1, 10.]


class TestFit_gauss3D_signal_size(TestSingle, unittest.TestCase):
    shape = (64, 64, 64)
    fit_function = 'gauss'
    var_size = True
    var_signal = True
    diameter = 21
    pos_err = 2
    p0_pos = [32.3, 32.9, 31.7]
    size_err_factor = [0.2, 5.]
    signal_err_factor = [0.1, 10.]


class TestFit_gauss3D_a_signal_size(TestSingle, unittest.TestCase):
    shape = (64, 64, 32)
    fit_function = 'gauss'
    var_size = True
    var_signal = True
    diameter = (21, 21, 15)
    pos_err = [2, 2, 1]
    p0_pos = [32.3, 32.9, 16.7]
    size_err_factor = [0.5, 2]
    signal_err_factor = [0.1, 10.]

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb'], exit=False)
