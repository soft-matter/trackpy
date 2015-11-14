import trackpy.packing
from trackpy.packing import pairCorrelationKDTree2D
import unittest

class TestPairCorrelation(unittest.TestCase):
	def setup(self):
		self.lattice = lattice()
		self.ring = rings()



	def latticeTest(self):
	
		#Calculate g_r on the center particle only (index 210)
		edges, g_r_one = pairCorrelationKDTree2D(self.lattice, dr=.1, cutoff=8, p_indexes=[210])
		g_r_one /= np.linalg.norm(g_r_one) #We care about the relative difference of g_r in this case, so let's normalize both.

		#Calculate g_r on all particles 
		edges, g_r_all = pairCorrelationKDTree2D(self.lattice, dr=.1, cutoff=8)
		g_r_all /= np.linalg.norm(g_r_all)

		#Assert the functions are essentially the same
		assert np.allclose(g_r_all, g_r_one, atol=.02)


	def ringTest(self):	
		edges, g_r = pairCorrelationKDTree2D(df, dr=.1, cutoff=10, p_indexes=[0], boundary = (-10,10,-10,10))		
		peaks = g_r[g_r > 0]

		assert len(peaks) == 9

		x = np.arange(1,10,1)
		r = peaks.max() * 1/x
	
		assert np.allclose(peaks, r)

	
	def lattice(self, n = 20):
		x,y = [],[]
		epsilon = 0.0
		for i in range(n):
			for j in range(n):
				x.append(i)
				y.append(j)

		return pandas.DataFrame({'x':x, 'y':y})


	def rings(self):
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