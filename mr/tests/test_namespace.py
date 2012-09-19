import unittest
import nose

class TestNamespace(unittest.TestCase):

    def test_api(self):
        try:
            from mr import api
        except ImportError:
            self.assert_(False)

if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vv', '-s', '-x'], exit=False)
