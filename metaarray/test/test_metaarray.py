import pytest
from pyarray.metaarray import metaarray, metadepend, metatime
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
energy = np.tile(energy[None,:], (npts,1))

TIME = metatime.MetaTime(t, name='time')
PHI = metadepend.MetaDepend(phi, name='phi')
THETA = metadepend.MetaDepend(theta, name='theta')
ENERGY = metadepend.MetaDepend(energy, x0=TIME, name='energy')

# Scalar, Vector, and Multi-dimensional time series data
s = np.random.rand(npts)
v = np.random.rand(npts, 3)
pad = np.random.rand(npts, nphi)
geo = np.random.rand(npts, nphi, ntheta)
dist = np.random.rand(npts, nphi, ntheta, nenergy)

S = metaarray.MetaArray(s, x0=TIME, name='scalar', cache=False)
V = metaarray.MetaArray(v, x0=TIME, name='vector', cache=False)
PAD = metaarray.MetaArray(pad, x0=TIME, x1=phi, name='PitchAngleDist', cache=False)
GEO = metaarray.MetaArray(geo, x0=TIME, x1=phi, x2=theta, name='GeoLocation', cache=False)
DIST = metaarray.MetaArray(dist, x0=TIME, x1=PHI, x2=THETA, x3=ENERGY, name='DistFunc', cache=False)

def test_scalar_with_scalar_index():
    assert S[0] == s[0]


def test_scalar_list_of_indices():
    x = S[[0,10,20,30]]
    assert (x == s[[0,10,20,30]]).all()
    assert (x.x0 == t[[0,10,20,30]]).all()


def test_scalar_with_index_range():
    x = S[0:50:10]
    assert (x == s[0:50:10]).all()
    assert (x.x0 == t[0:50:10]).all()


def test_scalar_with_scalar_time_index():
    assert S[t1] == s[0]


def test_scalar_with_list_of_times():
    x = S[[t1, t2]]
    assert (x == s[[0,-1]]).all()
    assert (x.x0 == t[[0,-1]]).all()


def test_scalar_with_time_range():
    x = S[t1:t2]
    assert (x == s).all()
    assert (x.x0 == t).all()


#
# Vectors
#
def test_vector_broadcast_with_scalar_index():
    assert (V[0] == v[0]).all()


def test_vector_broadcast_with_list_of_indices():
    x = V[[0,10,20,30]]
    assert (x == v[[0,10,20,30]]).all()
    assert (x.x0 == t[[0,10,20,30]]).all()


def test_vector_broadcast_with_slice():
    x = V[0:50:10]
    assert (x == v[0:50:10]).all()
    assert (x.x0 == t[0:50:10]).all()


def test_vector_with_scalar_index():
    x = V[0, :]
    assert (x == v[0, :]).all()
    assert not hasattr(x, 'x0')


def test_vector_with_list_of_indices():
    x = V[[0,10,20,30], :]
    assert (x == v[[0,10,20,30], :]).all()
    assert (x.x0 == t[[0,10,20,30]]).all()


def test_vector_with_slice():
    x = V[0:50:10, :]
    assert (x == v[0:50:10, :]).all()
    assert (x.x0 == t[0:50:10]).all()


def test_vector_broadcast_with_scalar_time():
    x = V[t1]
    assert (x == v[0]).all()
    assert not hasattr(x, 'x0')


def test_vector_broadcast_with_list_of_times():
    x = V[[t1, t2]]
    assert (x == v[[0,-1]]).all()
    assert (x.x0 == t[[0,-1]]).all()


def test_vector_broadcast_with_slice_of_times():
    x = V[t1:t2]
    assert (x == v).all()
    assert (x.x0 == t).all()


def test_vector_with_scalar_time():
    x = V[t1, :]
    assert (x == v[0, :]).all()
    assert not hasattr(x, 'x0')


def test_vector_with_list_of_times():
    x = V[[t1, t2], :]
    assert (x == v[[0,-1], :]).all()
    assert (x.x0 == t[[0,-1]]).all()


def test_vector_with_slice_of_times():
    x = V[t1:t2, :]
    assert (x == v).all()
    assert (x.x0 == t).all()


#
# Distribution
#
def test_dist_broadcast_with_scalar_index():
    x = DIST[0]
    assert (x == dist[0]).all()
    assert (x.x0 == phi).all()
    assert (x.x1 == theta).all()
    assert (x.x2 == energy).all()
    assert not hasattr(x.x2, 'x0')
    assert not hasattr(x, 'x3')
    
    x = DIST[:,0]
    assert (x == dist[:,0]).all()
    assert (x.x0 == t).all()
    assert (x.x1 == theta).all()
    assert (x.x2 == energy).all()
    assert (x.x2.x0 == t).all()
    assert not hasattr(x, 'x3')
    
    x = DIST[:,:,0]
    assert (x == dist[:,:,0]).all()
    assert (x.x0 == t).all()
    assert (x.x1 == phi).all()
    assert (x.x2 == energy).all()
    assert (x.x2.x0 == t).all()
    assert not hasattr(x, 'x3')

    x = DIST[:,:,:,0]
    assert (x == dist[:,:,:,0]).all()
    assert (x.x0 == t).all()
    assert (x.x1 == phi).all()
    assert (x.x2 == theta).all()
    assert not hasattr(x, 'x3')


#
# Other Cases
#
def test_none_slice_with_shallow_last_dim():
    '''
    This test case came about because MetaBase._depend_arrange() was
    stepping through the full set of indices but only the trimmed set
    of depended attribute names. The resulting error, "IndexError:
    list assignment index out of range" was caught by
    MetaArray.__getitem__() but passed over because other types of
    IndexErrors need to be caught and handled differently. 
    '''
    import datetime as dt
    import numpy as np
    from pyarray.metaarray import metabase, metaarray, metadepend, metatime

    n = 347724
    t1 = dt.datetime(2019, 12, 7, 3, 30, 30)
    t2 = dt.datetime(2019, 12, 7, 9, 53, 30)
    t = np.arange(t1, t2, (t2-t1)/n)
    data = np.random.rand(n, 4)
    TIME = metatime.MetaTime(t)
    DATA = metaarray.MetaArray(data, x0=TIME)

    x = DATA[slice(None, None, None), 0]
    assert (x == data[slice(None, None, None), 0]).all()


def test_index_has_time():
    # Also test datetime.datetime!
    assert metaarray.MetaArray._index_has_time(t1)
    assert metaarray.MetaArray._index_has_time([t1, t2])
    assert metaarray.MetaArray._index_has_time(t)
    assert metaarray.MetaArray._index_has_time(slice(t1, t2, None))
    assert metaarray.MetaArray._index_has_time((slice(t1, t2, None),
                                                slice(1, 10, 1),
                                                0,
                                                [1, 2, 3, 4]
                                                ))
    