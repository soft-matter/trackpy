import functools

import unittest
import numpy as np
from numpy.testing import assert_equal
import pandas

import trackpy
from trackpy import predict
from trackpy.tests.common import StrictTestCase
from trackpy.tests.test_find_link import CoordinateReader
from trackpy.utils import guess_pos_columns


def mkframe(n=1, Nside=3):
    xg, yg = np.mgrid[:Nside,:Nside] * 100
    dx = (n - 100)
    dy = -(n - 100)
    return pandas.DataFrame(
            dict(x=xg.flatten() + dx, y=yg.flatten() + dy, frame=n))

Nside_oversize = int(np.sqrt(100)) # Make subnet linker fail

class LinkWithPrediction:
    def get_linked_lengths(self, frames, pred, *args, **kw):
        """Track particles and return the length of each trajectory."""
        linked = self.link(frames, pred, *args, **kw)
        return linked.groupby('particle').x.count()

    def get_linked_lengths_from_iterfunc(self, frames, func, *args, **kw):
        """Track particles and return the length of each trajectory."""
        linked = pandas.concat(func(frames, *args, **kw), ignore_index=True)
        return linked.groupby('particle').x.count()

    def get_unwrapped_linker(self):
        return trackpy.link_df_iter


class LinkIterWithPrediction(LinkWithPrediction):
    def link(self, frames, pred, *args, **kw):
        # Takes an iterable of frames, and outputs a single linked DataFrame.
        defaults = {}
        defaults.update(kw)
        return pandas.concat(pred.link_df_iter(frames, *args, **defaults),
                             ignore_index=True)


class LinkDFWithPrediction(LinkWithPrediction):
    def link(self, frames, pred, *args, **kw):
        # Takes an iterable of frames, and outputs a single linked DataFrame.
        defaults = {}
        defaults.update(kw)
        return pred.link_df(pandas.concat(frames, ignore_index=True), *args, **defaults)


class BaselinePredictTests:
    def test_null_predict(self):
        """Make sure that a prediction of no motion does not interfere
        with normal tracking.
        """
        # link_df_iter
        pred = predict.NullPredict()
        ll = self.get_linked_lengths((mkframe(0), mkframe(25)),
                                pred, 45)
        assert all(ll.values == 2)

        # link_df
        pred = predict.NullPredict()
        ll_df = pred.link_df(pandas.concat((mkframe(0), mkframe(25))), 45)
        # print(ll_df)
        assert all(ll_df.groupby('particle').x.count().values == 2)

        # Make sure that keyword options are handled correctly
        # (This checks both link_df and link_df_iter)
        # The conditional is so we won't test find_link in this way,
        # due to the way column names are baked into the
        # artificial image code.
        if not getattr(self, 'coords_via_images', False):
            features = [mkframe(0), mkframe(25)]
            for f in features:
                f.rename(columns=lambda n: n + '_', inplace=True)
            pred = predict.NullPredict()
            ll_df = self.link(features, pred, 45, t_column='frame_',
                                          pos_columns=['x_', 'y_'])
            assert all(ll_df.groupby('particle').x_.count().values == 2)

    def test_predict_decorator(self):
        """Make sure that a prediction of no motion does not interfere
        with normal tracking.
        """
        pred = predict.null_predict
        pred_link = functools.partial(self.get_unwrapped_linker(), predictor=pred)
        ll = self.get_linked_lengths_from_iterfunc((mkframe(0), mkframe(25)),
                                pred_link, 45)
        assert all(ll.values == 2)

    def test_fail_predict(self):
        ll = self.get_linked_lengths_from_iterfunc((mkframe(0), mkframe(25), mkframe(65)),
                                self.get_unwrapped_linker(), 45)
        assert not all(ll.values == 2)

    def test_subnet_fail(self):
        with self.assertRaises(trackpy.SubnetOversizeException):
            Nside = Nside_oversize
            ll = self.get_linked_lengths_from_iterfunc((mkframe(0, Nside),
                                     mkframe(25, Nside),
                                     mkframe(75, Nside)),
                                    self.get_unwrapped_linker(), 100)


class BaselinePredictIterTests(LinkIterWithPrediction, BaselinePredictTests, StrictTestCase):
    pass


class BaselinePredictDFTests(LinkDFWithPrediction, BaselinePredictTests, StrictTestCase):
    pass



class VelocityPredictTests:
    def test_simple_predict(self):
        pred = self.predict_class()
        ll = self.get_linked_lengths((self.mkframe(0), self.mkframe(25), self.mkframe(65)),
                                pred, 45)
        assert all(ll.values == 3)

    def test_big_predict(self):
        Nside = Nside_oversize
        pred = self.predict_class()
        ll = self.get_linked_lengths((self.mkframe(0, Nside), self.mkframe(25, Nside),
                                 self.mkframe(75, Nside)),
                                pred, 45)
        assert all(ll.values == 3)

    def test_predict_newparticles(self):
        """New particles should get the velocities of nearby ones."""
        pred = self.predict_class()
        ll = self.get_linked_lengths((self.mkframe(0), self.mkframe(25), self.mkframe(65, 4),
                                self.mkframe(105, 4)),
                                pred, 45)
        assert not any(ll.values == 1)

    def test_predict_memory(self):
        pred = self.predict_class()
        frames = [self.mkframe(0), self.mkframe(25), self.mkframe(65),
                  self.mkframe(105), self.mkframe(145)]
        ll = self.get_linked_lengths(frames, pred, 45)
        assert all(ll.values == len(frames))

        # Knock out a particle. Make sure tracking fails.
        frames[3].loc[5, 'x'] = np.nan
        frames[3] = frames[3].dropna()
        pred = self.predict_class()
        tr = self.link(frames, pred, 45)
        starts = tr.groupby('particle').frame.min()
        ends = tr.groupby('particle').frame.max()
        assert not all(ends - starts == 145)

        pred = self.predict_class()
        tr = self.link(frames, pred, 45, memory=1)
        starts = tr.groupby('particle').frame.min()
        ends = tr.groupby('particle').frame.max()
        assert all(ends - starts == 145), 'Prediction with memory fails.'

    def test_predict_diagnostics(self):
        if getattr(self, 'coords_via_images', False):
            raise unittest.SkipTest
        """Minimally test predictor instrumentation."""
        pred = self.instrumented_predict_class()
        Nside = Nside_oversize
        frames = (self.mkframe(0, Nside), self.mkframe(25, Nside),
                  self.mkframe(75, Nside))
        ll = self.get_linked_lengths(frames, pred, 45)
        assert all(ll.values == 3)
        diags = pred.dump()
        assert len(diags) == 2
        for i, d in enumerate(diags):
            assert d['t1'] == frames[i+1].frame.iloc[0]
            assert 'state' in d
            assert np.all(d['particledf']['x_act'] == frames[i+1].x)


class NearestVelocityPredictTests(VelocityPredictTests):
    def setUp(self):
        self.predict_class = predict.NearestVelocityPredict
        self.instrumented_predict_class = \
            predict.instrumented()(self.predict_class)
        self.mkframe = mkframe

    def test_initial_guess(self):
        """When an accurate initial velocity is given, velocities
        in the first pair of frames may be large."""

        # Initializing a guess without specifying pos_columns should
        # raise an error
        with self.assertRaises(ValueError):
            pred = self.predict_class(
                initial_guess_positions=[(0., 0.)],
                initial_guess_vels=[(1, -1)])

        pred = self.predict_class(
            initial_guess_positions=[(0., 0.)],
            initial_guess_vels=[(-1, 1)],
            pos_columns=['y', 'x'])
        ll = self.get_linked_lengths((self.mkframe(0), self.mkframe(100),
                                 self.mkframe(200)),
                                pred, 45)
        assert all(ll.values == 3)

        # Since we can't specify nonstandard pos_columns for find_link,
        # skip the remaining checks if we're testing find_link
        if getattr(self, 'coords_via_images', False):
            return

        # Giving a pos_columns to the linker that conflicts with the one
        # used to initialize the predictor raises an error.
        pred = self.predict_class(
            initial_guess_positions=[(0., 0.)],
            initial_guess_vels=[(1, -1)],
            pos_columns=['x', 'y'])
        with self.assertRaises(ValueError):
            ll = self.get_linked_lengths((self.mkframe(0), self.mkframe(100),
                                          self.mkframe(200)),
                                         pred, 45, pos_columns=['y', 'x'])
        # Giving the same pos_columns is OK.
        pred = self.predict_class(
            initial_guess_positions=[(0., 0.)],
            initial_guess_vels=[(1, -1)],
            pos_columns=['x', 'y'])
        ll = self.get_linked_lengths((self.mkframe(0), self.mkframe(100),
                                      self.mkframe(200)),
                                     pred, 45, pos_columns=['x', 'y'])

    def test_pos_columns(self):
        """Test that pos_columns can be specified in any order, or not at all.

        Example by @freemansw1 and @snilsn, bug report by @wyu54.
        See https://github.com/soft-matter/trackpy/issues/699
        """
        # Two particles with increasing x, with opposite velocities in y.
        # They cross between frames 2 and 3.
        d = {'frame': [1, 2, 3, 4, 1, 2, 3, 4],
             'x': [0, 10, 20, 30, 10, 15, 20, 25],
             'y': [0, 10, 20, 30, 30, 20, 10, 0]}
        df = pandas.DataFrame(data=d)
        df_list = [frame for i, frame in df.groupby('frame')]

        def assert_correct(df):
            # The particle that starts at (0, 0) should finish at (30, 30)
            pid = df[(df.x == 0) & (df.y == 0)].particle.iloc[0]
            traj = df[df.particle == pid].set_index('frame')
            assert_equal(traj.x[4], 30)
            assert_equal(traj.y[4], 30)


        pred = self.predict_class()
        traj_1 = self.link(df_list, pred, 100)
        assert_correct(traj_1)

        pred = self.predict_class()
        traj_2 = self.link(df_list, pred, 100, pos_columns=['x', 'y'])
        assert_correct(traj_2)

        pred = self.predict_class()
        traj_3 = self.link(df_list, pred, 100, pos_columns=['y', 'x'])
        assert_correct(traj_3)



class NearestVelocityPredictIterTests(LinkIterWithPrediction, NearestVelocityPredictTests, StrictTestCase):
    pass


class NearestVelocityPredictDFTests(LinkDFWithPrediction, NearestVelocityPredictTests, StrictTestCase):
    pass


class DriftPredictTests(VelocityPredictTests):
    def setUp(self):
        self.predict_class = predict.DriftPredict
        self.instrumented_predict_class = \
            predict.instrumented()(self.predict_class)
        self.mkframe = mkframe

    def test_initial_guess(self):
        """When an accurate initial velocity is given, velocities
        in the first pair of frames may be large."""

        # Initializing a guess without specifying pos_columns should
        # raise an error
        with self.assertRaises(ValueError):
            pred = self.predict_class(initial_guess=(1, -1))

        # A bad initial guess will fail.
        pred = self.predict_class(initial_guess=(1, -1), pos_columns=['y', 'x'])
        ll = self.get_linked_lengths((self.mkframe(0), self.mkframe(100),
                                      self.mkframe(200)),
                                     pred, 45)
        assert not all(ll.values == 3)

        pred = self.predict_class(initial_guess=(-1, 1), pos_columns=['y', 'x'])
        ll = self.get_linked_lengths((self.mkframe(0), self.mkframe(100),
                                 self.mkframe(200)),
                                 pred, 45)
        assert all(ll.values == 3)

        # Since we can't specify nonstandard pos_columns for find_link,
        # skip the remaining check if we're testing find_link
        if getattr(self, 'coords_via_images', False):
            return

        # The guess is interpreted according to pos_columns
        pred = self.predict_class(initial_guess=(1, -1), pos_columns=['x', 'y'])
        ll = self.get_linked_lengths((self.mkframe(0), self.mkframe(100),
                                      self.mkframe(200)),
                                     pred, 45)
        assert all(ll.values == 3)


class DriftPredictIterTests(LinkIterWithPrediction, DriftPredictTests, StrictTestCase):
    pass


class DriftPredictDFTests(LinkDFWithPrediction, DriftPredictTests, StrictTestCase):
    pass


class ChannelPredictXTests(VelocityPredictTests):
    def setUp(self):
        self.predict_class = functools.partial(
            predict.ChannelPredict, 3, minsamples=3)
        self.instrumented_predict_class = functools.partial(
            predict.instrumented()(predict.ChannelPredict), 3, minsamples=3)
        self.mkframe = self._channel_frame

    def _channel_frame(self, n=1, Nside=3):
        xg, yg = np.mgrid[:Nside,:Nside] * 100
        dx = (n - 100) * np.sqrt(2)
        dy = 0.
        return pandas.DataFrame(
                dict(x=xg.flatten() + dx, y=yg.flatten() + dy, frame=n))

    def test_initial_guess(self):
        """When an accurate initial velocity profile is given, velocities
        in the first pair of frames may be large."""
        def _shear_frame(t=1., Nside=4):
            xg, yg = np.mgrid[:Nside,:Nside] * 100
            dx = 0.0045 * t * yg
            return pandas.DataFrame(
                dict(x=(xg + dx).flatten(), y=yg.flatten(), frame=t))

        inity = np.arange(4) * 100
        initprof = np.vstack((inity, inity*0.0045)).T
        # We need a weird bin size (110) to avoid bin boundaries coinciding
        # with particle positions.
        pred = predict.ChannelPredict(110, minsamples=3,
                                      initial_profile_guess=initprof)
        # print(_shear_frame(1.))
        ll = self.get_linked_lengths((_shear_frame(0), _shear_frame(100),
                                 _shear_frame(200), _shear_frame(300)),
                        pred, 45)
        assert all(ll.values == 4)


class ChannelPredictXIterTests(LinkIterWithPrediction, ChannelPredictXTests, StrictTestCase):
    pass


class ChannelPredictXDFTests(LinkDFWithPrediction, ChannelPredictXTests, StrictTestCase):
    pass


class ChannelPredictYTests(VelocityPredictTests):
    def setUp(self):
        self.predict_class = functools.partial(
            predict.ChannelPredict, 3, 'y', minsamples=3)
        self.instrumented_predict_class = functools.partial(
            predict.instrumented()(predict.ChannelPredict),
            3, 'y', minsamples=3)
        self.mkframe = self._channel_frame

    def _channel_frame(self, n=1, Nside=3):
        xg, yg = np.mgrid[:Nside,:Nside] * 100
        dx = 0.
        dy = (n - 100) * np.sqrt(2)
        return pandas.DataFrame(
            dict(x=xg.flatten() + dx, y=yg.flatten() + dy, frame=n))


class ChannelPredictYIterTests(LinkIterWithPrediction, ChannelPredictYTests, StrictTestCase):
    pass


class ChannelPredictYDFTests(LinkDFWithPrediction, ChannelPredictYTests, StrictTestCase):
    pass


# Test legacy linking functions, wrapped by a predictor method
class LegacyLinkWithPrediction:
    def get_linked_lengths(self, frames, pred, *args, **kw):
        """Track particles and return the length of each trajectory."""
        linked = self.link(frames, pred, *args, **kw)
        return linked.groupby('particle').x.count()

    def get_linked_lengths_from_iterfunc(self, frames, func, *args, **kw):
        """Track particles and return the length of each trajectory."""
        linked = pandas.concat(func(frames, *args, **kw), ignore_index=True)
        return linked.groupby('particle').x.count()

    def get_unwrapped_linker(self):
        return trackpy.linking.legacy.link_df_iter


class LegacyLinkIterWithPrediction(LegacyLinkWithPrediction):
    def link(self, frames, pred, *args, **kw):
        # Takes an iterable of frames, and outputs a single linked DataFrame.
        defaults = {}
        defaults.update(kw)
        wrapped_linker_iter = functools.partial(pred.wrap,
                                                trackpy.linking.legacy.link_df_iter)
        return pandas.concat(pred.link_df_iter(frames, *args, **defaults),
                             ignore_index=True)


class LegacyLinkDFWithPrediction(LegacyLinkWithPrediction):
    def link(self, frames, pred, *args, **kw):
        # Takes an iterable of frames, and outputs a single linked DataFrame.
        defaults = {}
        defaults.update(kw)
        wrapped_linker_df = functools.partial(pred.wrap_single,
                                              trackpy.linking.legacy.link_df_iter)
        return wrapped_linker_df(pandas.concat(frames, ignore_index=True), *args, **defaults)


class LLBaselinePredictIterTests(LegacyLinkIterWithPrediction, BaselinePredictTests, StrictTestCase):
    pass


class LLBaselinePredictDFTests(LegacyLinkDFWithPrediction, BaselinePredictTests, StrictTestCase):
    pass


class LLNearestVelocityPredictIterTests(LegacyLinkIterWithPrediction, NearestVelocityPredictTests, StrictTestCase):
    pass


class LLNearestVelocityPredictDFTests(LegacyLinkDFWithPrediction, NearestVelocityPredictTests, StrictTestCase):
    pass


class LLDriftPredictIterTests(LegacyLinkIterWithPrediction, DriftPredictTests, StrictTestCase):
    pass


class LLDriftPredictDFTests(LegacyLinkDFWithPrediction, DriftPredictTests, StrictTestCase):
    pass


class LLChannelPredictXIterTests(LegacyLinkIterWithPrediction, ChannelPredictXTests, StrictTestCase):
    pass


class LLChannelPredictXDFTests(LegacyLinkDFWithPrediction, ChannelPredictXTests, StrictTestCase):
    pass


class LLChannelPredictYIterTests(LegacyLinkIterWithPrediction, ChannelPredictYTests, StrictTestCase):
    pass


class LLChannelPredictYDFTests(LegacyLinkDFWithPrediction, ChannelPredictYTests, StrictTestCase):
    pass


## find_link prediction tests
# Define a mixins that convert a normal prediction test class into one
# that uses find_link.
class FindLinkWithPrediction(LinkWithPrediction):
    def setUp(self):
        super().setUp()
        self.linker_opts = dict(separation=10, diameter=15)
        # Disable certain tests that are redundant here
        # and would require more code to support.
        self.coords_via_images = True

    def get_unwrapped_linker(self):
        def link_iter(f, search_range, *args, **kw):
            kw = dict(self.linker_opts, **kw)
            size = 3
            separation = kw['separation']
            # convert the iterable to a single DataFrame (OK for tests)
            f = [_f for _f in f]
            indices = [_f['frame'].iloc[0] for _f in f]
            f = pandas.concat([_f for _f in f], ignore_index=True)
            topleft = (f[['y', 'x']].min().values - 4 * separation).astype(
                int)
            f[['y', 'x']] -= topleft
            shape = (f[['y', 'x']].max().values + 4 * separation).astype(
                int)
            reader = CoordinateReader(f, shape, size, t=indices)

            for i, frame in trackpy.find_link_iter(reader,
                                                   search_range=search_range,
                                                   *args, **kw):
                frame[['y', 'x']] += topleft
                yield frame
        return link_iter


# Just run one set of prediction tests, because there is no equivalent
# of pred.link_df in the API for find_link.
class FindLinkIterWithPrediction(FindLinkWithPrediction):
    def link(self, frames, pred, *args, **kw):
        # Takes an iterable of frames, and outputs a single linked DataFrame.
        defaults = {}
        defaults.update(kw)
        link_df_iter = self.get_linker_iter(pred)
        return pandas.concat(link_df_iter(frames, *args, **defaults),
                             ignore_index=True)

    def get_linker_iter(self, pred):
        def link_iter(f, search_range, *args, **kw):
            kw = dict(self.linker_opts, **kw)
            size = 3
            separation = kw['separation']
            # convert the iterable to a single DataFrame (OK for tests)
            f = [_f for _f in f]
            indices = [_f['frame'].iloc[0] for _f in f]
            f = pandas.concat([_f for _f in f], ignore_index=True)
            topleft = (f[['y', 'x']].min().values - 4 * separation).astype(
                int)
            f[['y', 'x']] -= topleft
            shape = (f[['y', 'x']].max().values + 4 * separation).astype(
                int)
            reader = CoordinateReader(f, shape, size, t=indices)

            # Ordinarily, pos_columns would be set when initializing the
            # predictor, or it would be automatically set by pred.link_df etc.
            # We must specify it before the predictor is given any
            # particle position data.
            if getattr(pred, 'pos_columns', None) is None:
                pred.pos_columns = guess_pos_columns(f)
            pred.t_column = kw.get('t_column', 'frame')

            # FindLinker uses image coordinates only, and therefore has no use
            # for pos_columns.
            if 'pos_columns' in kw:
                del kw['pos_columns']

            for i, frame in trackpy.find_link_iter(reader, predictor=pred.predict,
                                                   search_range=search_range,
                                                   *args, **kw):
                pred.observe(frame)
                frame[['y', 'x']] += topleft

                yield frame
        return link_iter


class FLBaselinePredictTests(FindLinkIterWithPrediction, BaselinePredictTests, StrictTestCase):
    pass


class FLNearestVelocityPredictTests(FindLinkIterWithPrediction, NearestVelocityPredictTests, StrictTestCase):
    pass


class FLDriftPredictTests(FindLinkIterWithPrediction, DriftPredictTests, StrictTestCase):
    pass


class FLChannelPredictXTests(FindLinkIterWithPrediction, ChannelPredictXTests, StrictTestCase):
    pass


class FLChannelPredictYTests(FindLinkIterWithPrediction, ChannelPredictYTests, StrictTestCase):
    pass


if __name__ == '__main__':
    import unittest
    unittest.main()
