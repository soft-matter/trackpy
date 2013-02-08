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
import trackpy.tracking as pt
import trackpy.identification as tid
import numpy as np
import random


def test_easy_tracking():
    level_count = 5
    p_count = 16
    levels = []

    for j in range(level_count):
        level = []
        for k in np.arange(p_count) * 2:
            level.append(pt.PointND(j, (j, k)))
        levels.append(level)

    hash_generator = lambda: pt.Hash_table((level_count + 1, p_count * 2 + 1), .5)
    tracks = pt.link(levels, 1.5, hash_generator)

    assert len(tracks) == p_count

    for t in tracks:
        x, y = zip(*[p.pos for p in t])
        dx = np.diff(x)
        dy = np.diff(y)

        assert np.sum(dx) == level_count - 1
        assert np.sum(dy) == 0


def test_iden():
    SEED = 314
    random.seed(SEED)

    X = range(10, 200, 20)
    Y = range(10, 200, 20)
    random.shuffle(X)
    random.shuffle(Y)
    img = tid.gen_fake_data(np.vstack([X, Y]), 5, 2.5, (210, 210))
    bp_img = tid.band_pass(img, 2, 2.5)

    res_lm = tid.find_local_max(bp_img, 3, .5)

    locs, mass, r2 = tid.subpixel_centroid(bp_img, res_lm, 3)
    assert np.all(np.abs(locs - res_lm) < .05)
