import unittest

import trackpy.diag

class DiagTests(unittest.TestCase):
    def test_performance_report(self):
        trackpy.diag.performance_report()
