from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import os
import logging
import warnings

import pims
import trackpy
import trackpy.diag
from trackpy.tests.common import StrictTestCase

path, _ = os.path.split(os.path.abspath(__file__))

class DiagTests(StrictTestCase):
    def test_performance_report(self):
        trackpy.diag.performance_report()

    def test_dependencies(self):
        trackpy.diag.dependencies()


class LoggerTests(StrictTestCase):
    def test_heirarchy(self):
        self.assertTrue(trackpy.linking.logger.parent is trackpy.logger)
        self.assertTrue(trackpy.feature.logger.parent is trackpy.logger)
        self.assertTrue(trackpy.preprocessing.logger.parent is trackpy.logger)

    def test_convenience_funcs(self):
        trackpy.quiet(True)
        self.assertEqual(trackpy.logger.level, logging.WARN)
        trackpy.quiet(False)
        self.assertEqual(trackpy.logger.level, logging.INFO)

        trackpy.ignore_logging()
        self.assertEqual(len(trackpy.logger.handlers), 0)
        self.assertEqual(trackpy.logger.level, logging.NOTSET)
        self.assertTrue(trackpy.logger.propagate)

        trackpy.handle_logging()
        self.assertEqual(len(trackpy.logger.handlers), 1)
        self.assertEqual(trackpy.logger.level, logging.INFO)
        self.assertEqual(trackpy.logger.propagate, 1)
