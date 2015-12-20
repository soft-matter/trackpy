
from trackpy.static import pairCorrelationKDTree2D, pairCorrelationKDTree3D
from trackpy.static import _points_ring3D
import unittest
import pandas
import numpy as np
import matplotlib.pyplot as plt


class TestPairCorrelation(unittest.TestCase):


    def test_correlation2D_lattice(self):
        ### Lattice Test
        # With proper edge handling, g(r) of the particle at the center should be the same as g(r) for all particles.
        lattice = self._lattice2D()

        # Calculate g_r on the center particle only (index 210)
        edges, g_r_one = pairCorrelationKDTree2D(lattice, dr=.1, cutoff=8, p_indices=[210])
        g_r_one /= np.linalg.norm(g_r_one) #We care about the relative difference of g_r in this case, so let's normalize both.

        # Calculate g_r on all particles
        edges, g_r_all = pairCorrelationKDTree2D(lattice, dr=.1, cutoff=8)
        g_r_all /= np.linalg.norm(g_r_all)

        # Calculate g_r on all particles
        edges, g_r_no_edge = pairCorrelationKDTree2D(lattice, dr=.1, cutoff=8, handle_edge=False)
        g_r_no_edge /= np.linalg.norm(g_r_no_edge)

        # Assert the functions are essentially the same
        self.assertTrue(np.allclose(g_r_all, g_r_one, atol=.02))

        # Turning off edge handling should give incorrect result
        self.assertFalse(np.allclose(g_r_all, g_r_no_edge, atol=.02))


    def test_correlation2D_ring(self):
        # Ring test
        # Generate a series of concentric shells, each with the same number of particles.
        # The peaks in g(r) should decay as 1/r.
        ring = self._rings2D()

        edges, g_r = pairCorrelationKDTree2D(ring, dr=.1, cutoff=10, p_indices=[0], boundary = (-10., 10., -10., 10.))
        g_r /= np.linalg.norm(g_r)
        peaks = g_r[g_r > 0]

        self.assertTrue( len(peaks) == 9 )

        x = np.arange(1,10,1)
        r = peaks.max() * 1/x

        self.assertTrue( np.allclose(peaks, r, atol=.01) )


    def test_correlation3D_ring(self):
        # Ring test
        # Generate a series of concentric shells, each with the same number of particles.
        # The peaks in g(r) should decay as 1/r^2.
        ring = self._rings3D()
        edges, g_r = pairCorrelationKDTree3D(ring, dr=.1, cutoff=10, p_indices=[len(ring) - 1], boundary = (-10., 10., -10., 10., -10., 10.), handle_edge=True)
        g_r /= np.linalg.norm(g_r)
        peaks = g_r[g_r > 0]
        assert len(peaks) == 9
        x = np.arange(1,10,1)
        r = peaks.max() * 1/x**2
        self.assertTrue( np.allclose(peaks, r, atol=.02) )


    def test_correlation3D_lattice(self):
        ### Lattice Test
        # With proper edge handling, g(r) of the particle at the center should be the same as g(r) for all particles.
        lattice = self._lattice3D(n = 20)

        # Calculate g_r on the center particle only (index 210)
        edges, g_r_one = pairCorrelationKDTree3D(lattice, dr=.1, cutoff=7, p_indices=[4649])
        g_r_one /= np.linalg.norm(g_r_one) #We care about the relative difference of g_r in this case, so let's normalize both.

        # Calculate g_r on all particles
        edges, g_r_all = pairCorrelationKDTree3D(lattice, dr=.1, cutoff=7)
        g_r_all /= np.linalg.norm(g_r_all)

        # Calculate g_r on all particles
        edges, g_r_no_edge = pairCorrelationKDTree3D(lattice, dr=.1, cutoff=7, handle_edge=False)
        g_r_no_edge /= np.linalg.norm(g_r_no_edge)

        # Assert the functions are essentially the same
        self.assertTrue(np.allclose(g_r_all, g_r_one, atol=.04))

        # Turning off edge handling should give incorrect result
        self.assertFalse(np.allclose(g_r_all, g_r_no_edge, atol=.04))

    def _lattice2D(self, n = 20):
        #Generates 2D lattice, spacing = 1
        x,y = [],[]
        epsilon = 0.0
        for i in range(n):
            for j in range(n):
                x.append(i)
                y.append(j)

        return pandas.DataFrame({'x':x, 'y':y})


    def _rings2D(self):
        #Generates concentric rings, with a particle at the center
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


    def _lattice3D(self, n = 20):
        #Generates 3D lattice, spacing = 1
        x,y,z = [],[],[]
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    x.append(i)
                    y.append(j)
                    z.append(k)

        return pandas.DataFrame({'x':x, 'y':y, 'z':z})


    def _rings3D(self):
        #Generates concentric spherical shells, with a particle at the center
        epsilon = .02
        r = np.arange(1, 10, 1) + epsilon
        refx, refy, refz = _points_ring3D(r, 0, 500)
        df = pandas.DataFrame({'x': np.concatenate(refx), 'y': np.concatenate(refy),
                               'z': np.concatenate(refz)})
        df.iloc[len(df)] = [0.,0.,0.]
        return df

if __name__ == '__main__':
    unittest.main()