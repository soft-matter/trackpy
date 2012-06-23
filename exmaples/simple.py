#Copyright 2012 Thomas A Caswell
#tcaswell@uchicago.edu
#http://jfi.uchicago.edu/~tcaswell
#
#This program is free software; you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation; either version 3 of the License, or (at
#your option) any later version.
#
#This program is distributed in the hope that it will be useful, but
#WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program; if not, see <http://www.gnu.org/licenses>.

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
			level.append(pt.PointND(0,np.asarray((j+2,k+2))+np.random.randn(2)/10))

# do the tracking
t = pt.link_full(levels,(20,20),.5,pt.Hash_table)

# plot tracks
fig = plt.figure()
ax = fig.gca()
for trk in t:
    x,y = zip(*[p.pos for p in trk.points])
    ax.plot(x,y)

    
plt.show()
