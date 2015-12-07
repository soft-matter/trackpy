
from trackpy.packing import pairCorrelationKDTree2D
import unittest
import pandas
import numpy as np
import matplotlib.pyplot as plt


class TestPairCorrelation(unittest.TestCase):
	

	def test_lattice(self):

		lattice = self._lattice()

		#Calculate g_r on the center particle only (index 210)
		edges, g_r_one = pairCorrelationKDTree2D(lattice, dr=.1, cutoff=8, p_indexes=[210])
		g_r_one /= np.linalg.norm(g_r_one) #We care about the relative difference of g_r in this case, so let's normalize both.


		#Calculate g_r on all particles 
		edges, g_r_all = pairCorrelationKDTree2D(lattice, dr=.1, cutoff=8)
		g_r_all /= np.linalg.norm(g_r_all)


		#Calculate g_r on all particles 
		edges, g_r_no_edge = pairCorrelationKDTree2D(lattice, dr=.1, cutoff=8, handle_edge=False)
		g_r_no_edge /= np.linalg.norm(g_r_no_edge)

		
		#Assert the functions are essentially the same
		self.assertTrue(np.allclose(g_r_all, g_r_one, atol=.02))

		#Turning off edge handling should give incorrect result
		self.assertFalse(np.allclose(g_r_all, g_r_no_edge, atol=.02))




	def test_ring(self):	
		ring = self._rings()

		edges, g_r = pairCorrelationKDTree2D(ring, dr=.1, cutoff=10, p_indexes=[0], boundary = (-10.,10.,-10.,10.))	
		g_r /= np.linalg.norm(g_r)	
		peaks = g_r[g_r > 0]

		self.assertTrue( len(peaks) == 9 )

		x = np.arange(1,10,1)
		r = peaks.max() * 1/x

		self.assertTrue( np.allclose(peaks, r, atol=.01) )



	def _lattice(self, n = 20):
		x,y = [],[]
		epsilon = 0.0
		for i in range(n):
			for j in range(n):
				x.append(i)
				y.append(j)

		return pandas.DataFrame({'x':x, 'y':y})


	def _rings(self):
		theta = np.linspace(0, 2*np.pi, 10)
		points = np.zeros((100,2))

		i = 0
		epsilon = .02
		for r in range(10):
			points[i:i+10, 0] = (r + epsilon)*np.cos(theta) 
			points[i:i+10, 1] = (r + epsilon)*np.sin(theta) 
			i += 10
		points[:10] = 0

		return pandas.DataFrame(points, columns = ['x', 'y'])


if __name__ == '__main__':
	unittest.main()