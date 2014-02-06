import unittest
import nose.tools
import numpy as np
import pandas
import trackpy
from trackpy import predict

def mkframe(n=1, Nside=3):
    xg, yg = np.mgrid[:Nside,:Nside]
    dx = (n - 1)
    dy = -(n - 1)
    return pandas.DataFrame(
            dict(x=xg.flatten() + dx, y=yg.flatten() + dy, frame=n))

def link(frames, linker, *args, **kw):
    defaults = {'neighbor_strategy': 'KDTree'}
    defaults.update(kw)
    return pandas.concat(linker(frames,  *args, **defaults),
                       ignore_index=True)

def get_linked_lengths(frames, linker, *args, **kw):
    """Track particles and return the length of each trajectory."""
    return link(frames, linker, *args, **kw).groupby('particle').x.count()

Nside_oversize = int(np.sqrt(100)) # Make subnet linker fail

class BaselinePredictTests(unittest.TestCase):
    def test_null_predict(self):
        """Make sure that a prediction of no motion does not interfere
        with normal tracking.
        """
        pred = predict.NullPredict()
        ll = get_linked_lengths((mkframe(0), mkframe(0.25)),
                                pred.link_df_iter, 0.45)
        assert all(ll.values == 2)

    def test_fail_predict(self):
        ll = get_linked_lengths((mkframe(0), mkframe(0.25), mkframe(0.65)),
                                trackpy.link_df_iter, 0.45)
        assert not all(ll.values == 2)

    @nose.tools.raises(trackpy.SubnetOversizeException)
    def test_subnet_fail(self):
        Nside = Nside_oversize
        ll = get_linked_lengths((mkframe(0, Nside),
                                 mkframe(0.25, Nside),
                                 mkframe(0.75, Nside)), trackpy.link_df_iter, 1)

class VelocityPredictTests(unittest.TestCase):
    def test_simple_predict(self):
        pred = predict.NearestVelocityPredict()
        ll = get_linked_lengths((mkframe(0), mkframe(0.25), mkframe(0.65)),
                                pred.link_df_iter, 0.45)
        assert all(ll.values == 3)

    def test_big_predict(self):
        Nside = Nside_oversize
        pred = predict.NearestVelocityPredict()
        ll = get_linked_lengths((mkframe(0, Nside), mkframe(0.25, Nside),
                                 mkframe(0.75, Nside)),
                                pred.link_df_iter, 0.45)
        assert all(ll.values == 3)

    def test_predict_newparticles(self):
        """New particles should get the velocities of nearby ones."""
        pred = predict.NearestVelocityPredict()
        ll = get_linked_lengths((mkframe(0), mkframe(0.25), mkframe(0.65, 4),
                                mkframe(1.05, 4)),
                                pred.link_df_iter, 0.45)
        assert not any(ll.values == 1)

    def test_predict_memory(self):
        pred = predict.NearestVelocityPredict()
        frames = [mkframe(0), mkframe(0.25), mkframe(0.65),
                  mkframe(1.05), mkframe(1.45)]
        ll = get_linked_lengths(frames, pred.link_df_iter, 0.45)
        assert all(ll.values == len(frames))

        # Knock out a particle. Make sure tracking fails.
        frames[3].x[5] = np.nan
        frames[3] = frames[3].dropna()
        pred = predict.NearestVelocityPredict()
        tr = link(frames, pred.link_df_iter, 0.45)
        starts = tr.groupby('particle').frame.min()
        ends = tr.groupby('particle').frame.max()
        assert not all(ends - starts == 1.45)

        pred = predict.NearestVelocityPredict()
        tr = link(frames, pred.link_df_iter, 0.45, memory=1)
        starts = tr.groupby('particle').frame.min()
        ends = tr.groupby('particle').frame.max()
        assert all(ends - starts == 1.45), 'Prediction with memory fails.'

if __name__ == '__main__':
    unittest.main()
