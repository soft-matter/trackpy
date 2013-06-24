# Copyright 2012 Daniel B. Allan
# dallan@pha.jhu.edu, daniel.b.allan@gmail.com
# http://pha.jhu.edu/~dallan
# http://www.danallan.com
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or (at
# your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses>.

import collections
import functools
import re
from datetime import datetime, timedelta

def fit_powerlaw(data, plot=True):
    """Fit a powerlaw by doing a linear regression in log space."""
    ys = DataFrame(data)
    x = Series(data.index.values, index=data.index, dtype=np.float64)
    values = DataFrame(index=['n', 'A'])
    fits = {}
    for col in ys:
        y = ys[col].dropna()
        slope, intercept, r, p, stderr = \
            stats.linregress(np.log(x), np.log(y))
        print 'slope', slope, ', intercept', intercept
        values[col] = [slope, np.exp(intercept)]
        fits[col] = x.apply(lambda x: np.exp(intercept)*x**slope)
    values = values.T
    fits = pd.concat(fits, axis=1)
    if plot:
        import plots
        plots.fit(data, fits, logx=True, logy=True)
    return values

class memo(object):
   """Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   http://wiki.python.org/moin/PythonDecoratorLibrary#Memoize """
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value
   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__
   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)

def extract(pattern, string, group, convert=None):
    """Extract a pattern from a string. Optionally, convert it
    to a desired type (float, timestamp, etc.) by specifying a function.
    When the pattern is not found, gracefully return None."""
    # group may be 1, (1,) or (1, 2).
    if type(group) is int:
        grp = (group,)
    elif type(group) is tuple:
        grp = group
    assert type(grp) is tuple, "The arg 'group' should be an int or a tuple."
    try:
        result = re.search(pattern, string, re.DOTALL).group(*grp)
    except AttributeError:
        # For easy unpacking, when a tuple is expected, return a tuple of Nones.
        return None if type(group) is int else (None,)*len(group)
    return convert(result) if convert else result

def timestamp(ts_string):
    "Convert a timestamp string to a datetime type."
    if ts_string is None: return None
    return datetime.strptime(ts_string, '%Y-%m-%d %H:%M:%S')

def time_interval(raw):
    "Convert a time interval string into a timedelta type."
    if raw is None: return None
    m = re.match('([0-9][0-9]):([0-5][0-9]):([0-5][0-9])', raw)
    h, m, s = map(int, m.group(1,2,3))
    return timedelta(hours=h, minutes=m, seconds=s)
