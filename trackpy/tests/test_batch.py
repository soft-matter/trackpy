from trackpy import batch, locate
from pandas import DataFrame
from PIL import Image
import tempfile
import os
import numpy as np

import pytest
from unittest.mock import patch, MagicMock
import shutil
import contextlib

from typing import Iterable

@contextlib.contextmanager
def temporary_directory():
    """
    Context manager which removes temporary directory when required.
    """

    tempdir = tempfile.mkdtemp()
    try:
        yield tempdir
    finally:
        shutil.rmtree(tempdir)

def save_and_retrieve_frames_with_pillow(frames:Iterable[np.typing.NDArray[np.float64]],
                            extension:str):
    """
    Helper for saving frames in a temporary directory and retrieving these using the pillow Image object.
    """
    retrieved_frames = []

    with temporary_directory() as tempdir:
        for i in range(len(frames)):
            img = Image.fromarray(frames[i])
            img.save(os.path.join(tempdir, f"image{i}{extension}"))

        for filename in os.listdir(tempdir):
            img_from_file = Image.open(os.path.join(tempdir, filename), mode="r")
            retrieved_frames.append(np.array(img_from_file))
            img_from_file.close()

    return retrieved_frames

@patch("trackpy.feature.locate", return_value = DataFrame(
                {'x':[5],
                 'y':[5],
                 'particle':[1]})) # mock 'locate' function with some fixed output
class TestSingleThread:
    """
    Test batch functionality which does not depend on multiprocessing.
    """
    input_frames = [np.zeros((10, 10))*10]

    @patch("trackpy.utils.Pool") # have to mock pool to check that it isn't used
    def test_does_not_use_multiprocessing_pool(self, mocked_pool:MagicMock, mocked_locate:MagicMock):
        batch(self.input_frames, diameter=1, processes=1)

        mocked_pool.assert_not_called()
 
    def test_frames_are_numbered_starting_from_one(self, mocked_locate:MagicMock):
        tracks: DataFrame = batch(self.input_frames, 1, processes=1)

        assert [actual == expected for actual, expected 
                in zip(tracks['frame'], range(10))]

    def test_frames_from_file_are_numbered_starting_from_one(self, mocked_locate:MagicMock):
        frames_from_tiff = save_and_retrieve_frames_with_pillow(self.input_frames, ".tiff")
        
        tracks: DataFrame = batch(frames_from_tiff, 1, processes=1)

        assert [actual == expected for actual, expected 
                in zip(tracks['frame'], range(10))]

@patch("trackpy.utils.Pool") # Have to mock Pool where it's used, which is trackpy.utils in this case
class TestMultiprocessing:
    """
    Test multiprocessing configuration using mocked Pool instance.
    """

    # Define some dummy input and output to mock the 'locate' function.
    # The content of the frames doesn't matter; this is just to test the batch logic.
    input_frames = [np.zeros((10, 10)),]*10
    pool_output = [DataFrame({'x':[5,], 'y':[5,],'particle':[0]}) for _ in range(len(input_frames))]

    def test_uses_multiprocessing_pool_for_more_than_one_process(self, mocked_pool:MagicMock):
        instance = mocked_pool.return_value
        instance.imap.return_value = self.pool_output

        batch(self.input_frames, diameter=1, processes=8)

        mocked_pool.assert_called_once_with(processes=8)

    @patch('trackpy.feature.partial')
    def test_partial_function_uses_locate(self,
                                          mocked_partial:MagicMock,
                                          mocked_pool:MagicMock):
        """
        Test that partial function used in multiprocessing is locate function with indicated kwargs.
        """
        instance = mocked_pool.return_value
        instance.imap.return_value = self.pool_output
        kwargs = {'diameter':11, 'invert':False, 'minmass':500}

        batch(self.input_frames, **kwargs, processes=8)

        mocked_partial.assert_called_once_with(locate, spiff=False, **kwargs)

    @patch('trackpy.feature.partial')
    def test_partial_function_is_used_in_multiprocessing(self, mocked_partial:MagicMock,
                                                        mocked_pool:MagicMock):
        def expected_partial_func():
            pass

        instance = mocked_pool.return_value
        instance.imap.return_value = self.pool_output
        mocked_partial.return_value = expected_partial_func

        batch(self.input_frames, diameter=1, processes=8)

        instance.imap.assert_called_once_with(expected_partial_func, self.input_frames)

    def test_frames_from_file_are_numbered_starting_from_one(self, mocked_pool:MagicMock):
        instance = mocked_pool.return_value
        instance.imap.return_value = self.pool_output
        frames_from_tiff = save_and_retrieve_frames_with_pillow(self.input_frames, ".tiff")
        
        tracks: DataFrame = batch(frames_from_tiff, 1, processes=8)

        assert [actual == expected for actual, expected 
                in zip(tracks['frame'], range(10))]