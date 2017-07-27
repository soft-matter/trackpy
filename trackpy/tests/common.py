from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
import trackpy as tp
from trackpy.utils import cKDTree, pandas_sort, make_pandas_strict


class StrictTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Suppress logging messages
        tp.quiet()
        # Catch attempts to set values on an inadvertent copy of a Pandas object.
        make_pandas_strict()
        # Make numpy strict
        np.seterr('raise')


def sort_positions(actual, expected):
    tree = cKDTree(actual)
    deviations, argsort = tree.query([expected])
    if len(set(range(len(actual))) - set(argsort[0])) > 0:
        raise AssertionError("Position sorting failed. At least one feature is "
                             "very far from where it should be.")
    return deviations, actual[argsort][0]


def assert_coordinates_close(actual, expected, atol):
    assert_equal(len(actual), len(expected))
    _, sorted_actual = sort_positions(actual, expected)
    assert_allclose(sorted_actual, expected, atol=atol)


def assert_traj_equal(actual, expected, pos_atol=1):
    assert_equal(len(actual), len(expected))
    actual = pandas_sort(actual, 'frame').reset_index(drop=True)
    expected = pandas_sort(expected, 'frame').reset_index(drop=True)
    actual_order = []
    for frame_no in expected['frame'].unique():
        actual_f = actual[actual['frame'] == frame_no]
        expected_f = expected[expected['frame'] == frame_no]
        assert_equal(len(actual_f), len(expected_f),
                     err_msg='Actual and expected numbers of features '
                             'differ in frame %i' % frame_no)
        tree = cKDTree(actual_f[['y', 'x']].values)
        devs, argsort = tree.query(expected_f[['y', 'x']].values)
        assert_allclose(devs, 0., atol=pos_atol)
        actual_order.extend(actual_f.index[argsort].tolist())

    actual = actual.loc[actual_order].reset_index(drop=True, inplace=False)
    for p_actual in actual.particle.unique():
        actual_ind = actual.index[actual['particle'] == p_actual]
        p_expected = expected.loc[actual_ind[0], 'particle']
        expected_ind = expected.index[expected['particle'] == p_expected]
        assert_array_equal(actual_ind, expected_ind,
                           err_msg='Actual and expected linking results '
                           'differ for actual particle %i/expected particle %i'
                           '' % (p_actual, p_expected))

