from __future__ import division
import matplotlib
matplotlib.use('qt4agg')
import trackpy.tracking as pt
import matplotlib.pyplot as plt
import numpy as np

# generate fake data
levels = []
# 15 planes
for i in range(15):
    level = []
    # add the current level to the list of levels
    levels.append(level)
    # a 15 by 15 grid
    for j in range(15):
        for k in range(15):
            # displace the location from the grid by a guassian with width 1/10
            level.append(pt.PointND(i, np.asarray((j + 2, k + 2)) + np.random.randn(2) / 10))

# do the tracking
hash_generator = lambda: pt.Hash_table((20, 20), .5)
t = pt.link(levels, .75, hash_generator)

# plot tracks
fig = plt.figure()
ax = fig.gca()
for trk in t:
    x, y = zip(*[p.pos for p in trk.points])
    ax.plot(x, y)


plt.show()
