import operator

import streams
from streams import Stream
from dask.utils import M
import pandas as pd


class Streaming(object):
    _subtype = object
    def __init__(self, stream=None, example=None, columns=None):
        if columns is not None and example is None:
            example = pd.DataFrame({c: [] for c in columns})
        assert example is not None
        self.example = example
        self.stream = stream or Stream()

    def map_partitions(self, func, *args, **kwargs):
        example = func(self.example, *args, **kwargs)
        stream = self.stream.map(func, *args, **kwargs)

        if isinstance(example, pd.DataFrame):
            return StreamingDataFrame(stream, example)
        elif isinstance(example, pd.Series):
            return StreamingSeries(stream, example)
        else:
            return StreamingScalar(stream, example)

    def accumulate_partitions(self, func, *args, **kwargs):
        start = kwargs.pop('start', streams.core.no_default)
        returns_state = kwargs.pop('returns_state', False)
        example = func(start, self.example, *args, **kwargs)
        stream = self.stream.accumulate(func, *args, start=start,
                returns_state=returns_state, **kwargs)

        if isinstance(example, pd.DataFrame):
            return StreamingDataFrame(stream, example)
        elif isinstance(example, pd.Series):
            return StreamingSeries(stream, example)
        else:
            return Streaming(stream, example)

    def __add__(self, other):
        return self.map_partitions(operator.add, other)

    def __mul__(self, other):
        return self.map_partitions(operator.mul, other)

    def emit(self, x):
        self.verify(x)
        self.stream.emit(x)

    def verify(self, x):
        if not isinstance(x, self._subtype):
            raise TypeError("Expected type %s, got type %s" %
                            (self._subtype, type(x)))


class StreamingFrame(Streaming):
    def sum(self):
        def update(accumulator, new):
            return accumulator + new.sum()
        return self.accumulate_partitions(update, start=0)


class StreamingDataFrame(StreamingFrame):
    _subtype = pd.DataFrame
    @property
    def columns(self):
        return self.example.columns

    def __getitem__(self, index):
        return self.map_partitions(operator.getitem, index)

    def __getattr__(self, key):
        if key in self.columns:
            return self.map_partitions(getattr, key)
        else:
            raise AttributeError("StreamingDataFrame has no attribute %r" % key)

    def __dir__(self):
        o = set(dir(type(self)))
        o.update(self.__dict__)
        o.update(c for c in self.columns if
                 (isinstance(c, pd.compat.string_types) and
                 pd.compat.isidentifier(c)))
        return list(o)

    def verify(self, x):
        super(StreamingDataFrame, self).verify(x)
        if list(x.columns) != list(self.example.columns):
            raise IndexError("Input expected to have columns %s, got %s" %
                             (self.example.columns, x.columns))

    def mean(self):
        start = pd.DataFrame({'sums': 0, 'counts': 0},
                             index=self.example.columns)
        return self.accumulate_partitions(_accumulate_mean, start=start,
                                          returns_state=True)


class StreamingSeries(StreamingFrame):
    _subtype = pd.Series

    def mean(self):
        start = pd.Series({'sums': 0, 'counts': 0})
        return self.accumulate_partitions(_accumulate_mean, start=start,
                                          returns_state=True)



def _accumulate_mean(accumulator, new):
    accumulator = accumulator.copy()
    accumulator['sums'] += new.sum()
    accumulator['counts'] += new.count()
    result = accumulator['sums'] / accumulator['counts']
    return accumulator, result

