#Copyright 2013 Thomas A Caswell
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

import trackpy.tracking as pt
import trackpy.identification as tid
import matplotlib.pyplot as plt

import numpy as np

img = tid.gen_fake_data(np.vstack([np.arange(10, 200, 20),
                                   np.arange(10, 200, 20)]), 5, 2.5, (210, 210))
bp_img = tid.band_pass(img, 2, 2.5)


res_lm = tid.find_local_max(bp_img, 3, .5)

locs, mass, r2 = tid.subpixel_centroid(bp_img, res_lm, 3)

# make figure
fig = plt.figure()
ax = fig.gca()
# display image
ax.imshow(bp_img, cmap='gray', interpolation='nearest')

# add pixval like output
ax.format_coord = lambda x, y: 'r=%d,c=%d,v=%0.2f' % (int(x + .5),
                                                      int(y + .5),
                                                      bp_img[int(x + .5), int(y + .5)] if
                                                      int(x + .5) < bp_img.shape[0] and int(y + .5) < bp_img.shape[1] else 0)

ax.plot(*locs, linestyle='none', marker='o')
plt.show()
