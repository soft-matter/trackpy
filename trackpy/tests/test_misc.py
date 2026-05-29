import logging
import os
import types
import unittest
import warnings

import trackpy
import trackpy.diag
from trackpy.tests.common import StrictTestCase
from trackpy.try_numba import NUMBA_AVAILABLE

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


class NumbaTests(StrictTestCase):
    def setUp(self):
        if not NUMBA_AVAILABLE:
            raise unittest.SkipTest("Numba not installed. Skipping.")
        self.funcs = trackpy.try_numba._registered_functions

    def tearDown(self):
        if NUMBA_AVAILABLE:
            trackpy.enable_numba()

    def test_registered_numba_functions(self):
        self.assertGreater(len(self.funcs), 0)

    def test_enabled(self):
        trackpy.enable_numba()
        for registered_func in self.funcs:
            module = __import__(registered_func.module_name, fromlist='.')
            func = getattr(module, registered_func.func_name)
            self.assertIs(func, registered_func.compiled)
            self.assertNotIsInstance(func, types.FunctionType)

    def test_disabled(self):
        trackpy.disable_numba()
        for registered_func in self.funcs:
            module = __import__(registered_func.module_name, fromlist='.')
            func = getattr(module, registered_func.func_name)
            self.assertIs(func, registered_func.ordinary)
            self.assertIsInstance(func, types.FunctionType)
