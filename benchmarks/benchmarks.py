from vbench.api import Benchmark, BenchmarkRunner
from datetime import datetime

common_setup = """
import mr
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

def random_walk(N):
    return np.cumsum(np.random.randn(N))
"""

setup = common_setup + """

def draw_gaussian_spot(image, pos, r):
    assert image.shape[0] != image.shape[1], \
        "For stupid numpy broadcasting reasons, don't make the image square."
    x, y = np.meshgrid(*np.array(map(np.arange, image.shape)) - pos)
    max_value = np.iinfo(image.dtype).max
    spot = max_value*np.exp(-(x**2 + y**2)/r).T
    image += spot

def gen_random_locations(shape, count):
    np.random.seed(0)
    return np.array([map(np.random.randint, shape) for _ in xrange(count)])

def draw_spots(shape, locations, r):
    image = np.zeros(shape, dtype='uint8')
    for x in locations:
        draw_gaussian_spot(image, x, r)
    return image

SHAPE = (1200, 1000)
COUNT = 10
R = 7
locations = gen_random_locations(SHAPE, COUNT)
img = draw_spots(SHAPE, locations, R)
for module in ['mr', 'mr.feature', 'mr.core.feature']:
    try:
        locate = __import__(module).locate
    except ImportError:
        continue
    except AttributeError:
        continue
    else:
        break
"""

locate_artificial_sparse = Benchmark("locate(img, 7)", setup, ncalls=10,
    name='locate_artificial_sparse')

setup = common_setup + """
# One 1D stepper
N = 500
f = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})

"""

link_one_continuous_stepper = Benchmark("mr.link(f, 5)", setup, ncalls=5,
    name='link_one_continuous_stepper')

link_trackpy_one_continuous_stepper = Benchmark("mr.link_trackpy(f, 5)", 
     setup, ncalls=5, name='link_trackpy_one_continuous_stepper')

setup = common_setup + """
N = 500
Y = 2
# Begin second feature one frame later than the first, so the probe labeling (0, 1) is
# established and not arbitrary.
a = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1), 
              'frame': np.arange(1, N)})
f = pd.concat([a, b]).sort('frame')
"""

link_two_nearby_steppers = Benchmark("mr.link(f, 5)", setup, ncalls=5,
    name='link_two_nearby_steppers')

link_trackpy_two_nearby_steppers = Benchmark("mr.link_trackpy(f, 5)", 
     setup, ncalls=5, name='link_trackpy_two_nearby_steppers')

setup = common_setup + """
np.random.seed(0)
N = 100 
Y = 250
M = 50 # margin, because negative values raise OutOfHash
initial_positions = [(10, 11), (10, 18), (14, 15), (20, 21), (13, 13),
                     (10, 10), (17, 19)]
import itertools
c = itertools.count()
def walk(x, y):
    i = next(c)
    return DataFrame({'x': M + x + random_walk(N - i), 
                      'y': M + y + random_walk(N - i),
                     'frame': np.arange(i, N)})
f = pd.concat([walk(*pos) for pos in initial_positions])
"""

link_nearby_continuous_random_walks = Benchmark("mr.link(f, 5)", setup,
    ncalls=5, name='link_nearby_continuous_random_walks')

link_trackpy_nearby_continuous_random_walks = \
    Benchmark("mr.link_trackpy(f, 5)", setup,
    ncalls=5, name='link_nearby_continuous_random_walks')
