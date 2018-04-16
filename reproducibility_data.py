"""
This script generates the data for the tests in test_reproducibilty.
"""
import os

import trackpy as tp
import numpy as np
import pims

version = 'VERSION'  # adjust this

pos_columns = ['y', 'x']
char_columns = ['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep']
testpath = os.path.join(os.path.dirname(tp.__file__), 'tests')
impath = os.path.join(testpath, 'video', 'image_sequence', '*.png')
npzpath = os.path.join(testpath, 'data',
                       'reproducibility_v{}.npz'.format(version))

v = pims.ImageSequence(impath)
# take reader that provides uint8!
assert np.issubdtype(v.dtype, np.uint8)
v0 = tp.invert_image(v[0])
v0_bp = tp.bandpass(v0, lshort=1, llong=9)
expected_find = tp.grey_dilation(v0, separation=9)
expected_find_bandpass = tp.grey_dilation(v0_bp, separation=9)
expected_refine = tp.refine_com(v0, v0_bp, radius=4,
                                coords=expected_find_bandpass)
expected_refine = expected_refine[expected_refine['mass'] >= 140]
expected_refine_coords = expected_refine[pos_columns].values
expected_locate = tp.locate(v0, diameter=9, minmass=140)
expected_locate_coords = expected_locate[pos_columns].values
df = tp.locate(v0, diameter=9)
df = df[(df['x'] < 64) & (df['y'] < 64)]
expected_characterize = df[pos_columns + char_columns].values

f = tp.batch(tp.invert_image(v), 9, minmass=140)
f_crop = f[(f['x'] < 320) & (f['x'] > 280) & (f['y'] < 280) & (f['x'] > 240)]
f_linked = tp.link(f_crop, search_range=5, memory=0)
f_linked_memory = tp.link(f_crop, search_range=5, memory=2)
link_coords = f_linked[pos_columns + ['frame']].values
expected_linked = f_linked['particle'].values
expected_linked_memory = f_linked_memory['particle'].values

np.savez_compressed(npzpath, expected_find, expected_find_bandpass,
                    expected_refine_coords, expected_locate_coords,
                    link_coords, expected_linked, expected_linked_memory,
                    expected_characterize)
