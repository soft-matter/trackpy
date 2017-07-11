from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import os
import numpy as np
import pandas as pd
import pims
import nose
from numpy.testing import assert_allclose

import trackpy as tp
from trackpy.preprocessing import invert_image
from trackpy.utils import cKDTree
from trackpy.tests.common import assert_traj_equal, StrictTestCase

path, _ = os.path.split(os.path.abspath(__file__))
reproduce_fn = os.path.join(path, 'data',
                            'reproduce_{}.csv'.format(tp.__version__))


def test_pos_equal(actual, expected, pos_atol=0.001, lost_atol=1):
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
        super(TestReproducibility, cls).setUpClass()
        # generate a new file
        video = pims.ImageSequence(
            os.path.join(path, 'video', 'image_sequence'))
        actual = tp.batch(invert_image(video), diameter=9, minmass=240)
        actual = tp.link_df(actual, search_range=5, memory=2)
        actual.to_csv(reproduce_fn)

    @classmethod
    def tearDownClass(cls):
        super(TestReproducibility, cls).tearDownClass()
        os.remove(reproduce_fn)

    def setUp(self):
        self.expected = pd.read_csv(os.path.join(path, 'data',
                                                 'reproduce_reference.csv'))
        self.actual = pd.read_csv(reproduce_fn)
        self.compared = test_pos_equal(self.actual, self.expected,
                                       pos_atol=0.001, lost_atol=1)
        self.characterize_rtol = 0.0001

    def test_find(self):
        n_lost = len(self.compared[0])
        self.assertEqual(n_lost, 0,
                         "{0} of {1} features were not found.".format(
                             n_lost, len(self.expected)))
        n_appeared = len(self.compared[1])
        self.assertEqual(n_appeared, 0,
                         "{0} of {1} features were found unexpectedly.".format(
                             n_appeared, len(self.actual)))

    def test_refine(self):
        n_dev = len(self.compared[2][0])
        self.assertEqual(n_dev, 0,
                         "{0} of {1} features have moved more than the tolerance.".format(
                             n_dev, len(self.actual)))

    def test_characterize(self):
        equal = self.compared[3]
        equal_f = self.expected.iloc[equal[0]].reset_index(drop=True), \
                  self.actual.iloc[equal[1]].reset_index(drop=True)

        for field in ['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep']:
            assert_allclose(equal_f[0][field].values,
                            equal_f[1][field].values,
                            rtol=self.characterize_rtol)

    def test_link(self):
        # run the linking on the expected coordinates, so that tests are
        # independent of possible different refine or find results
        actual = tp.link_df(self.expected, search_range=5, memory=0)
        assert_traj_equal(actual, self.expected)
