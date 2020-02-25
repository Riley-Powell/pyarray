import pytest
from pyarray import metaarray
import numpy as np

# Time interval
t1 = np.datetime64('2015-12-06T00:00:00')
t2 = np.datetime64('2015-12-06T10:09:33')
dt = np.timedelta64(5, 's')
t = np.arange(t1, t2, dt, dtype='datetime64[ms]')
npts = len(t)
nphi = 32
ntheta = 16
nenergy = 32

# Dependent data
phi = np.arange(0, 360, 360/nphi)
theta = np.arange(0, 180, 180/ntheta)
energy = np.logspace(np.log10(10), np.log10(40000), num=nenergy)
energy = np.tile(energy[None,:], (3,1))

PHI = metaarray.MetaDepend(phi, axis=1)
THETA = metaarray.MetaDepend(theta, axis=2)
ENERGY = metaarray.MetaDepend(energy, x0=t, axis=3)

# Scalar, Vector, and Multi-dimensional time series data
s = np.random.rand(npts)
pad = np.random.rand(npts, nphi)
geo = np.random.rand(npts, nphi, ntheta)
dist = np.random.rand(npts, nphi, ntheta, nenergy)

S = metaarray.MetaArray(s, t, name='scalar', cache=False)
PAD = metaarray.MetaArray(pad, x0=t, x1=phi, name='PitchAngleDist', cache=False)
GEO = metaarray.MetaArray(geo, x0=t, x1=phi, x2=theta, name='GeoLocation', cache=False)
DIST = metaarray.MetaArray(geo, x0=t, x1=phi, x2=theta, name='DistFunc', cache=False)


def test_scalar_with_scalar_index():
    assert S[0] == s[0]


def test_scalar_list_of_indices():
    x = S[[0,10,20,30]]
    assert all(x == s[[0,10,20,30]])
    assert all(x.x0 == t[[0,10,20,30]])


def test_scalar_with_index_range():
    x = S[0:50:10]
    assert all(x == s[0:50:10])
    assert all(x.x0 == t[0:50:10])


def test_scalar_with_scalar_time_index():
    assert S[t1] == s[0]


def test_scalar_with_list_of_times():
    x = S[[t1, t2]]
    assert all(x == s[[0,-1]])
    assert all(x.x0 == t[[0,-1]])


def test_scalar_with_time_range():
    x = S[t1:t2]
    assert all(x == s)
    assert all(x.x0 == t)


def test_vector_with_scalar_index():
    assert all(V[0] == v[0])