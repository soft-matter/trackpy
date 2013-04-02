from PIL import Image
import numpy as np

def tif_frames(filename):
    tif = Image.open(filename)
    end = False
    while not end:
        img = np.array(tif.getdata()).reshape(tif.size[::-1])
        try:
            tif.seek(1 + tif.tell())
        except EOFError:
            end = True
        yield img
