import collections
import trackpy as tp
import pandas as pd
import numpy as np

class Trajectories(object):
    """

    Parameters
    ----------
    data : an iterator or a callable that produces an iterator

    """
    def __init__(self, data, pos_columns=['x', 'y']):
        self.pos_columns = pos_columns
        self.data = data
        if callable(data):
            self.get_data = self._get_iterator
        else:
            # Make a plain DataFrame behave like chunks, like with item.
            self.get_data = lambda: iter(list([self.data]))

    def _get_iterator(self):
        chunks = self.data() # should return a generator
        if not isinstance(chunks, collections.Iterable):
            raise TypeError(
                "Callable data must return an iterator.")
        return chunks 

    def compute_drift(self, smoothing=0, query=None):
        if query is None:
            compute_drift = lambda chunk: tp.compute_drift(chunk, smoothing)
        else:
            compute_drift = lambda chunk: (
                tp.compute_drift(chunk.query(query), smoothing))

        chunks = self.get_data()
        drift = []
        last_value = 0
        drift_chunks = (compute_drift(chunk) for chunk in chunks)
        for chunk in drift_chunks:
            drift.append(chunk + last_value)
            if len(chunk) > 0:  # catch unusual case where chunk is empty
                last_value += chunk.iloc[-1]
        drift = pd.concat(drift)
        return drift

    def regional_drift(self, smoothing=0, divisions=None):
        if divisions is None:
            # Detect quadrant regions
            chunks = self.get_data()
            size = np.zeros(len(self.pos_columns))
            for chunk in chunks:
                size = np.maximum(size, chunk[self.pos_columns].max())
            divisions = {col: s // 2 for col, s in zip(self.pos_columns, size)}
            print 'Divided quadrants at', divisions
 
        subregions = {}

        for k, v in divisions.iteritems():
            drift = self.compute_drift(smoothing, query='%s > %d' % (k, v))
            subregions[k + '_plus'] = drift[k]
            drift = self.compute_drift(smoothing, query='%s < %d' % (k, v))
            subregions[k + '_minus'] = drift[k]

        return pd.concat(subregions, 1)

    def coverage(self):
        chunks = self.get_data()
        N = []
        for chunk in chunks:
            N.append(chunk.groupby('frame')['probe'].count())
        return pd.concat(N)

    def lengths(self, plot=True):
        chunks = self.get_data()
        cum_L = pd.DataFrame({'frame': []})
        for chunk in chunks:
            L = pd.DataFrame({'frame': chunk.groupby('probe')['frame'].count()})
            cum_L = cum_L.add(L, fill_value=0)
        if plot:
            cum_L.hist()
        return cum_L.squeeze()
