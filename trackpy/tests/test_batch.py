from trackpy import batch, locate
from pandas import DataFrame
from PIL import Image

from unittest.mock import patch, MagicMock

@patch('trackpy.utils.Pool') # must replace Pool where it's used, in this case trackpy.utils
class TestMultiprocessing:
    """
    Test multiprocessing configuration using mocked Pool instance.
    """

    # Define some dummy input and output to mock the 'locate' function.
    # The content of the frames doesn't matter; this is just to test the batch logic.
    input_frames = [Image.new("L", (10, 10))]
    output_tracks = [DataFrame({'x':[5,], 'y':[5,],'particle':[0]}),]

    def test_can_use_multiprocessing_pool(self, mocked_pool:MagicMock):
        instance = mocked_pool.return_value
        instance.imap.return_value = self.output_tracks

        batch(self.input_frames, diameter=1, processes=8)

        mocked_pool.assert_called_once_with(processes=8)

    @patch('trackpy.feature.partial')
    def test_partial_function_uses_locate(self, mocked_partial:MagicMock,
                                                        mocked_pool:MagicMock):
        instance = mocked_pool.return_value
        instance.imap.return_value = [DataFrame({'x':[5,], 'y':[5,],'particle':[0]})]

        batch(self.input_frames, diameter=1, processes=8)

        mocked_partial.assert_called_once_with(locate, diameter=1, spiff=False)

    @patch('trackpy.feature.partial')
    def test_partial_function_is_used_in_multiprocessing(self, mocked_partial:MagicMock,
                                                        mocked_pool:MagicMock):
        def expected_partial_func():
            pass

        instance = mocked_pool.return_value
        instance.imap.return_value = self.output_tracks
        mocked_partial.return_value = expected_partial_func

        batch(self.input_frames, diameter=1, processes=8)

        instance.imap.assert_called_once_with(expected_partial_func, self.input_frames)
