import os
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_allclose
from scipy.spatial import cKDTree

import trackpy as tp
from trackpy.preprocessing import invert_image
from trackpy.tests.common import TrackpyImageSequence
from trackpy.tests.common import assert_traj_equal, StrictTestCase

path, _ = os.path.split(os.path.abspath(__file__))


reproduce_fn = os.path.join(path, 'data', 'reproducibility_v0.4.npz')


def compare_pos_df(actual, expected, pos_atol=0.001, lost_atol=1):
    """Returns indices of equal and different positions inside dataframes
    `actual` and `expected`."""
    lost0 = []
    appeared1 = []
    dev0 = []
    dev1 = []
    equal0 = []
    equal1 = []
    for frame_no, expected_frame in expected.groupby('frame'):
        coords0 = expected_frame[['y', 'x']].values
        actual_frame = actual[actual['frame'] == frame_no]
        coords1 = actual_frame[['y', 'x']].values

        # use a KDTree to find nearest neighbors
        tree = cKDTree(coords1)
        devs, inds = tree.query(coords0)  # find nearest neighbors

        i_lost0 = np.argwhere(devs > lost_atol).ravel()
        # features that are equal
        i_equal0 = np.argwhere(devs < pos_atol).ravel()
        i_equal1 = inds[i_equal0]
        # features that are the same, but deviate in position
        i_dev0 = np.argwhere((devs < lost_atol) & (devs >= pos_atol)).ravel()
        i_dev1 = inds[i_dev0]
        # features that present in f1 and not in f0
        i_appeared1 = np.argwhere(~np.in1d(np.arange(len(coords1)),
                                           np.concatenate(
                                               [i_equal0, i_dev0]))).ravel()
        lost0.append(expected_frame.iloc[i_lost0].index.values)
        appeared1.append(actual_frame.iloc[i_appeared1].index.values)
        dev0.append(expected_frame.iloc[i_dev0].index.values)
        dev1.append(actual_frame.iloc[i_dev1].index.values)
        equal0.append(expected_frame.iloc[i_equal0].index.values)
        equal1.append(actual_frame.iloc[i_equal1].index.values)

    return np.concatenate(lost0), np.concatenate(appeared1), \
           (np.concatenate(dev0), np.concatenate(dev1)), \
           (np.concatenate(equal0), np.concatenate(equal1)),


class TestReproducibility(StrictTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        npz = np.load(reproduce_fn)
        cls.expected_find_raw = npz['arr_0']
        cls.expected_find_bp = npz['arr_1']
        cls.expected_refine = npz['arr_2']
        cls.expected_locate = npz['arr_3']
        cls.coords_link = npz['arr_4']
        cls.expected_link = npz['arr_5']
        cls.expected_link_memory = npz['arr_6']
        cls.expected_characterize = npz['arr_7']

        cls.v = TrackpyImageSequence(os.path.join(path, 'video',
                                                  'image_sequence', '*.png'))
        cls.v0_inverted = invert_image(cls.v[0])

    def setUp(self):
        self.diameter = 9
        self.minmass = 140
        self.memory = 2
        self.bandpass_params = dict(lshort=1, llong=self.diameter)
        self.find_params = dict(separation=self.diameter)
        self.refine_params = dict(radius=int(self.diameter // 2))
        self.locate_params = dict(diameter=self.diameter, minmass=self.minmass,
                                  characterize=False)
        self.link_params = dict(search_range=5)
        self.characterize_params = dict(diameter=self.diameter,
                                        characterize=True)
        self.pos_columns = ['y', 'x']
        self.char_columns = ['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep']

    def test_find_raw(self):
        actual = tp.grey_dilation(self.v0_inverted, **self.find_params)
        assert_array_equal(actual, self.expected_find_raw)

    def test_find_bp(self):
        image_bp = tp.bandpass(self.v0_inverted, **self.bandpass_params)
        actual = tp.grey_dilation(image_bp, **self.find_params)
        assert_array_equal(actual, self.expected_find_bp)

    def test_refine(self):
        coords_v0 = self.expected_find_bp
        image_bp = tp.bandpass(self.v0_inverted, **self.bandpass_params)
        df = tp.refine_com(self.v0_inverted, image_bp, coords=coords_v0,
                           **self.refine_params)
        actual = df[df['mass'] >= self.minmass][self.pos_columns].values

        assert_allclose(actual, self.expected_refine)

    def test_locate(self):
        df = tp.locate(self.v0_inverted, **self.locate_params)
        actual = df[self.pos_columns].values
        assert_allclose(actual, self.expected_locate)

    def test_link_nomemory(self):
        expected = pd.DataFrame(self.coords_link,
                                columns=self.pos_columns + ['frame'])
        expected['frame'] = expected['frame'].astype(int)
        actual = tp.link(expected, **self.link_params)
        expected['particle'] = self.expected_link

        assert_traj_equal(actual, expected)

    def test_link_memory(self):
        expected = pd.DataFrame(self.coords_link,
                                columns=self.pos_columns + ['frame'])
        expected['frame'] = expected['frame'].astype(int)
        actual = tp.link(expected, memory=self.memory, **self.link_params)
        expected['particle'] = self.expected_link_memory

        assert_traj_equal(actual, expected)

    def test_characterize(self):
        df = tp.locate(self.v0_inverted, diameter=9)
        df = df[(df['x'] < 64) & (df['y'] < 64)]
        actual_coords = df[self.pos_columns].values
        actual_char = df[self.char_columns].values

        try:
            assert_allclose(actual_coords,
                            self.expected_characterize[:, :2])
        except AssertionError:
            raise AssertionError('The characterize tests failed as the coords'
                                 ' found by locate were not reproduced.')
        assert_allclose(actual_char,
                        self.expected_characterize[:, 2:])
