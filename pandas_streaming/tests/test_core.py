from pandas_streaming import StreamingDataFrame, StreamingSeries
import pytest
from dask.dataframe.utils import assert_eq
import pandas as pd


def test_identity():
    sdf = StreamingDataFrame(columns=['x', 'y'])
    L = sdf.stream.sink_to_list()

    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    sdf.emit(df)

    assert L[0] is df
    assert list(sdf.example.columns) == ['x', 'y']

    x = sdf.x
    assert isinstance(x, StreamingSeries)
    L2 = x.stream.sink_to_list()
    assert not L2

    sdf.emit(df)
    assert isinstance(L2[0], pd.Series)
    assert assert_eq(L2[0], df.x)


def test_exceptions():
    sdf = StreamingDataFrame(columns=['x', 'y'])
    with pytest.raises(TypeError):
        sdf.emit(1)

    with pytest.raises(IndexError):
        sdf.emit(pd.DataFrame())


def test_sum():
    sdf = StreamingDataFrame(columns=['x', 'y'])
    df_out = sdf.sum().stream.sink_to_list()

    x = sdf.x
    x_out = x.sum().stream.sink_to_list()

    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    sdf.emit(df)
    sdf.emit(df)

    assert assert_eq(df_out[0], df.sum())
    assert assert_eq(df_out[1], df.sum() + df.sum())

    assert x_out[0] == df.x.sum()
    assert x_out[1] == df.x.sum() + df.x.sum()


def test_mean():
    sdf = StreamingDataFrame(columns=['x', 'y'])
    mean = sdf.mean()
    assert isinstance(mean, StreamingSeries)
    df_out = mean.stream.sink_to_list()


    x = sdf.x
    x_out = x.mean().stream.sink_to_list()

    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    sdf.emit(df)
    sdf.emit(df)

    assert assert_eq(df_out[0], df.mean())
    assert assert_eq(df_out[1], df.mean())

    assert x_out[0] == df.x.mean()
    assert x_out[1] == df.x.mean()


def test_arithmetic():
    a = StreamingDataFrame(columns=['x', 'y'])
    b = a + 1

    L1 = b.stream.sink_to_list()

    c = b.x * 10

    L2 = c.stream.sink_to_list()

    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    a.emit(df)

    assert assert_eq(L1[0], df + 1)
    assert assert_eq(L2[0], (df + 1).x * 10)


@pytest.mark.xfail(reason='need to zip two streaming dataframes together')
def test_pair_arithmetic():
    df = pd.DataFrame({'x': list(range(10)), 'y': [1] * 10})

    a = StreamingDataFrame(example=df.iloc[0])
    L = ((a.x + a.y) * 2).sink_to_list()

    a.emit(df.iloc[:5])
    a.emit(df.iloc[5:])

    assert len(L) == 2
    assert_eq(pd.concat(L, axis=0), (df.x + df.y) * 2)


def test_groupby_sum():
    df = pd.DataFrame({'x': list(range(10)), 'y': [1] * 10})

    a = StreamingDataFrame(example=df.iloc[:0])

    L1 = a.groupby(a.x % 3).y.sum().stream.sink_to_list()
    L2 = a.groupby(a.x % 3).sum().stream.sink_to_list()

    a.emit(df.iloc[:3])
    a.emit(df.iloc[3:7])
    a.emit(df.iloc[7:])

    assert assert_eq(L1[-1], df.groupby(df.x % 3).y.sum())
    assert assert_eq(L2[-1], df.groupby(df.x % 3).sum())
