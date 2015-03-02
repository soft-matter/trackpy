from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import os
import unittest
import warnings

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
        with warnings.catch_warnings(True) as w:
            warnings.simplefilter('always')
            _ = trackpy.ImageSequence(os.path.join(path, 'video/image_sequence/*.png'))
            assert len(w) == 1
