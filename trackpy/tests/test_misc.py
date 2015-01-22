from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import unittest

import trackpy.diag


class DiagTests(unittest.TestCase):
    def test_performance_report(self):
        trackpy.diag.performance_report()

    def test_dependencies(self):
        trackpy.diag.dependencies()
