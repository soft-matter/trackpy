import nose
import unittest
import os
import numpy as np
from scipy import ndimage
import trackpy.wire
from numpy.testing import assert_almost_equal

def curpath():
    path, _ = os.path.split(os.path.abspath(__file__))
    return path

def orientation(cov):
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvec = eigvecs[eigvals.argmax()]
    return np.arctan2(eigvec[1], eigvec[0])

class TestWire(unittest.TestCase):
   def setUp(self):
       self.oblique = np.load(os.path.join(curpath(), 'wire', 'oblique_frame.npy'))
       self.vertical = np.load(os.path.join(curpath(), 'wire', 'vertical_frame.npy'))
       self.horizontal = np.load(os.path.join(curpath(), 'wire', 'horizontal_frame.npy'))
       
   def test_oblique_wire(self):
       assert_almost_equal(trackpy.wire.tracking.analyze(self.oblique), 53.392, decimal=0)

   def test_vertical_wire(self):
       assert_almost_equal(trackpy.wire.tracking.analyze(self.vertical), 91.484, decimal=0)

   def test_horizontal_wire(self):
       assert_almost_equal(trackpy.wire.tracking.analyze(self.horizontal), -177.515, decimal=0)
if __name__ == '__main__':
    import nose
    nose.runmodule(argv=[__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
                   exit=False)
