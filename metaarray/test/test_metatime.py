import pytest
from metaarray import metatime
import numpy as np

# Test data
t0 = np.datetime64('2013-01-01T12:30:00')
t1 = t0 + np.timedelta64(2400, 'ms')
t2 = t0 + np.timedelta64(8600, 'ms')
duration = np.timedelta64(10)
tn = t0 + duration
t = metatime.MetaTime(np.arange(t0, tn, np.timedelta64(1, 's')))

def test_scalar_index():
    assert t[0] == t0


def test_nearest_item():
    assert t[t1] == np.datetime64('2013-01-01T12:30:02')


def test_range_slice():
    assert all(t[t1:t2] == np.arange(
                                t0+np.timedelta64(3, 's'),
                                t0+np.timedelta64(9, 's'),
                                np.timedelta64(1, 's')
                                )
               )


def test_nearest_list():
    assert all(t[[t1, t2]] == [np.datetime64('2013-01-01T12:30:02'),
                               np.datetime64('2013-01-01T12:30:09')])


def test_nearest_index_error():
    with pytest.raises(IndexError):
        t[t1,t2]


def test_tdelta_total():
    dt = t - t[0]
    assert(dt.total_seconds() == duration)
