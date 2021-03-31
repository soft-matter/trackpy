import os, glob
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
import trackpy as tp
from scipy.spatial import cKDTree
from trackpy.utils import pandas_sort, make_pandas_strict
from trackpy.artificial import SimulatedImage
from matplotlib.pyplot import imread


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


class TrackpyFrame(np.ndarray):
    """Simplified version of pims.Frame, for testing purposes only."""
    def __new__(cls, input_array, frame_no=None):
        # get a view of the input data as a Frame object
        obj = np.asarray(input_array).view(cls)
        obj.frame_no = frame_no
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.frame_no = getattr(obj, 'frame_no', None)

    def __array_wrap__(self, out_arr, context=None):
        # Handle scalars so as not to break ndimage.
        # See http://stackoverflow.com/a/794812/1221924
        if out_arr.ndim == 0:
            return out_arr[()]

        return np.ndarray.__array_wrap__(self, out_arr, context)


class TrackpyImageSequence:
    """Simplified version of pims.ImageSequence, for testing purposes only."""
    def __init__(self, path_spec):
        self._get_files(path_spec)

        tmp = imread(self._filepaths[0])
        self._first_frame_shape = tmp.shape

    def _get_files(self, path_spec):
        # deal with if input is _not_ a string
        if not isinstance(path_spec, str):
            # assume it is iterable and off we go!
            self._filepaths = list(path_spec)
            self._count = len(self._filepaths)
            return

        self.pathname = os.path.abspath(path_spec)  # used by __repr__
        if os.path.isdir(path_spec):
            directory = path_spec
            filenames = os.listdir(directory)
            make_full_path = lambda filename: (
                os.path.abspath(os.path.join(directory, filename)))
            filepaths = list(map(make_full_path, filenames))
        else:
            filepaths = glob.glob(path_spec)
        self._filepaths = list(sorted(filepaths))
        self._count = len(self._filepaths)

        # If there were no matches, this was probably a user typo.
        if self._count == 0:
            raise OSError("No files were found matching that path.")

    def __getitem__(self, j):
        image = imread(self._filepaths[j])
        image = (image * 255).astype(np.uint8)
        return TrackpyFrame(image, frame_no=j)

    def __len__(self):
        return self._count

    @property
    def frame_shape(self):
        return self._first_frame_shape

    @property
    def dtype(self):
        return np.uint8


class CoordinateReader:
    """Generate a pims.FramesSquence-like object that draws features at
    given coordinates"""
    def __init__(self, f, shape, size, t=None, **kwargs):
        self._f = f.copy()
        self.pos_columns = ['z', 'y', 'x'][-len(shape):]
        self.shape = shape
        self.size = size
        self.kwargs = kwargs
        self.im = SimulatedImage(shape, size, **self.kwargs)
        if t is None:
            self._len = int(f['frame'].max() + 1)
            self._inds = range(self._len)
        else:
            self._len = len(t)
            self._inds = t

    def __len__(self):
        return self._len

    def __iter__(self):
        # this is actually a hack to get find_link working with float-typed indices
        return (self.get_frame(i) for i in self._inds)

    def __getitem__(self, key):
        return self.get_frame(key)

    def get_frame(self, ind):
        self.im.clear()
        pos = self._f.loc[self._f['frame'] == ind, self.pos_columns].values
        for _pos in pos:
            self.im.draw_feature(_pos)
        return TrackpyFrame(self.im(), frame_no=ind)

    @property
    def pixel_type(self):
        return self.im.dtype

    @property
    def frame_shape(self):
        return self.im.shape
