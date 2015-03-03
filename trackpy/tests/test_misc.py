from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import os
import unittest
import warnings

import pims
import trackpy
import trackpy.diag

path, _ = os.path.split(os.path.abspath(__file__))

class DiagTests(unittest.TestCase):
    def test_performance_report(self):
        trackpy.diag.performance_report()

    def test_dependencies(self):
        trackpy.diag.dependencies()


class APITests(unittest.TestCase):
    def test_pims_deprecation(self):
        """Using a pims class should work, but generate a warning.

        The inclusion of these classes (and therefore this test) in
        trackpy is deprecated as of v0.3 and will be removed in a future
        version."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always', UserWarning)
            imseq = trackpy.ImageSequence(os.path.join(path, 'video/image_sequence/*.png'))
            assert isinstance(imseq, pims.ImageSequence)
            if len(w) != 1:
                print('Caught warnings:')
                for wrn in w:
                    print(wrn, wrn.message)
            assert len(w) == 1
