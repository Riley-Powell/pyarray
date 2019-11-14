# Instructions for subclassing numpy.ndarray are taken from (9/8/2017)
#   https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
import numpy as np
import pyarray
import pdb

class MetaArray(pyarray.MetaBase):
    
    def __new__(cls, x, x0=None, **kwargs):
        
        # Create a numpy array of the data
        obj = super(MetaArray, cls).__new__(cls, x, **kwargs)
        
        # Add new attributes
        if x0 is not None:
            obj.x0 = pyarray.MetaTime(x0)
        
        # Finally, we must return the newly created object:
        return obj

    
    def __array_finalize__(self, obj):
        if obj is None: return
        
        super(MetaArray, self).__array_finalize__(obj)
        self.x0 = getattr(obj, 'x0', None)
    
    
    def __getitem__(self, idx):
        try:
            obj = super(MetaArray, self).__getitem__(idx)
        
        # IndexError raised when idx is a datetime64 object
        # TypeError raised when slice contains datetime64 object
        except (TypeError, IndexError):
            try:
                index = self.x0.get_item_index(idx)
            
            # ValueError when idx is a tuple of indices
            except ValueError:
                index = self.x0.get_item_index(idx[0])
            
            except:
                raise
            
            return self[index]
        except:
            raise
        
        # X0
        if np.isscalar(idx):
            if np.ndim(self) > 1:
                obj = obj[None, ...]
                obj.x0 = self.x0[idx]
        
        elif type(idx) in (list, slice):
            obj.x0 = self.x0[idx]
        
        elif type(idx) is tuple:
            try:
                obj.x0 = self.x0[idx[0]]
            
            # IndexError if self.x0 is a scalar numpy.datetime64
            except IndexError:
                pass
        
        pdb.set_trace()
        
        # x1
        if hasattr(self, 'x1'):
            if np.isscalar(idx) | (type(idx) in (list, slice)):
                try:
                    obj.x1 = self.x1[idx, :]
                # IndexError idx was broadcast to all other dimensions
                except IndexError:
                    pass
            
            elif isinstance(idx, tuple):
                try:
                    obj.x1 = self.x1[idx[0], idx[1]]
                # IndexError if x1 is not record varying (i.e. 1D)
                except IndexError:
                    obj.x1 = self.x1[idx[1]]
        
        # x2
        if hasattr(self, 'x2'):
            if np.isscalar(idx) | (type(idx) in (list, slice)):
                try:
                    obj.x2 = self.x2[idx, :]
                # IndexError idx was broadcast to all other dimensions
                except IndexError:
                    pass
            
            elif isinstance(idx, tuple):
                try:
                    obj.x2 = self.x2[idx[0], idx[2]]
                # IndexError if x2 is not record varying (i.e. 1D)
                except IndexError:
                    obj.x2 = self.x2[idx[2]]
        
        return obj
        
        # Scalar index given
        #   - Scalars are stripped of times
#         if np.isscalar(idx):
#             try:
#                 obj = super(MetaArray, self).__getitem__(idx)
#                 if np.ndim(self) > 1:
#                     obj = obj[None, ...]
#                     obj.x0 = self.x0[idx]
#                 return obj
#             
#             except IndexError:
#                 index = self.x0.get_item_index(idx)
#                 return self[index]
#         
#         # Try to access as if indices were given
#         try:
#             obj = super(MetaArray, self).__getitem__(idx)
#             
#             
#             try:
#                 obj.x0 = self.x0.__getitem__(idx)
#             
#             # If idx is a tuple with idexes into multiple dimensions
#             # If all elements of idx are scalars, obj is a scalar
#             # and views cannot be returned.
#             except IndexError:
#                 pdb.set_trace()
#                 if np.isscalar(obj):
#                     pass
#                 else:
#                     raise
#             except AttributeError:
#                 pass
#             
#             return obj
#        
#        # Try to access as if times were given
#        except (TypeError, IndexError) as e:
#            pass
#        
#        indices = self.x0.get_item_index(idx)
#        if len(idx) > 1:
#            indices = (indices, *idx[1:])
#        
#        return self[indices]
#        
#        # TODO: Nearest-neighbor interpolation if dt is a time
#         try:
#             obj = super(MetaArray, self).__getitem__(dt)
#         except (TypeError, IndexError) as e:
#             raise
#         
#         # it must be datetime
#         # make sure it is a numpy datetime
#         dt = np.datetime64(dt,'s')
#         
#         # Return exact match
#         idx = dt == self.t
#         if np.any(idx):
#             return self[idx][0]
#         
#         # Nearest neighbor interpolation
#         idx = np.argwhere(dt < self.t)
#         idx = idx[0][0]
#         x_lo = self[idx - 1, ...]
#         x_hi = self[idx, ...]
#         t_lo = self.t[idx - 1]
#         t_hi = self.t[idx]
#         
#         return x_lo + (x_hi - x_lo) * ((dt - t_lo) / (t_hi - t_lo))
    
    
    def __check_time(self, t):
        return np.all(self.t == t)


def test_indexing():
    x = [[1,2,3]]
    x0 = np.datetime64('2015-12-06T00:23:04')
    data = metaarray.MetaArray(x, x0=x0)


def test_scalar_get_item():
    # Time interval
    t1 = np.datetime64('2015-12-06T00:00:00')
    t2 = np.datetime64('2015-12-06T10:09:33')
    dt = np.timedelta64(5, 's')
    t = np.arange(t1, t2, dt, dtype='datetime64[ms]')
    
    # Scalar, Vector, and Multi-dimensional time series data
    n = MetaArray(np.random.rand(len(t)), t, name='density', cache=True)
#    V = np.random.rand(len(t), 3)
#    E = np.random.rand(len(t), 32, 32)
    
    # Sub-interval for indexing
    t1 = np.datetime64('2015-12-06T00:23:04')
    t2 = np.datetime64('2015-12-06T00:25:34')
    
    # Scalar index
    n0 = n[0]
    print('n0={0:7.4f}'.format(n0))
    
    # List of indices
    nlist = n[[0,10,20,30]]
    print('x0=({0},), nlist=({1},)'.format(nlist.x0.size, nlist.size))
    
    # Index range
    nrange = n[0:50:10]
    print('x0=({0},), nrange=({1},)'.format(nrange.x0.size, nrange.size))
    
    # Scalar time
    nt0 = n[t1]
    print('nt0={0:7.4f}'.format(nt0))
    
    # List of times
    ntlist = n[[t1, t2]]
    print('x0=({0},), ntlist=({1},)'.format(ntlist.x0.size, ntlist.size))
    
    # Range of times
    ntrange = n[t1:t2]
    print('x0=({0},), ntrange=({1},)'.format(ntrange.x0.size, ntrange.size))


def test_vector_get_item():
    # Time interval
    t1 = np.datetime64('2015-12-06T00:00:00')
    t2 = np.datetime64('2015-12-06T10:09:33')
    dt = np.timedelta64(5, 's')
    t = np.arange(t1, t2, dt, dtype='datetime64[ms]')
    npts = len(t)
    
    # Scalar, Vector, and Multi-dimensional time series data
    V = MetaArray(np.random.rand(npts, 3), t, name='velocity', cache=False)
    
    # Sub-interval for indexing
    t1 = np.datetime64('2015-12-06T00:23:04')
    t2 = np.datetime64('2015-12-06T00:25:34')
    
    print('V.shape = {0}, type(V) = {1}'.format(V.shape, type(V)))
    
    # Broadcast scalar index
    subarray = V[0]
    print('Broadcase Scalar Index: V[0]')
    print('  x0={0}, subarray={1}'.format(subarray.x0.shape, subarray.shape))
    
    # Broadcast list of indices
    subarray = V[[0,10,20,30]]
    print('Broadcase List of Indices: V[[...]]')
    print('  x0={0}, subarray={1}'.format(subarray.x0.shape, subarray.shape))
    
    # Broadcast index range
    subarray = V[0:50:10]
    print('Broadcase Index Range: V[start:stop:step]')
    print('  x0={0}, subarray={1}'.format(subarray.x0.shape, subarray.shape))
    
    # Scalar index
    subarray = V[0, :]
    print('Scalar Index: V[0, :]')
    print('  x0={0}, subarray={1}'.format(subarray.x0.shape, subarray.shape))
    
    # List of indices
    subarray = V[[0,10,20,30], :]
    print('List of Indices: V[[...], :]')
    print('  x0={0}, subarray={1}'.format(subarray.x0.shape, subarray.shape))
    
    # Range of indices
    subarray = V[0:50:10, :]
    print('Range of Indices: V[start:stop:step, :]')
    print('  x0={0}, subarray={1}'.format(subarray.x0.shape, subarray.shape))
    
    # Broadcast scalar time
    subarray = V[t1]
    print('Broadcast Scalar Time: V[t1]')
    print('  x0={0}, subarray={1}'.format(subarray.x0.shape, subarray.shape))
    
    # Broadcast list of times
    subarray = V[[t1, t2]]
    print('Broadcast List of Times: V[[t1,t2]]')
    print('  x0={0}, subarray={1}'.format(subarray.x0.shape, subarray.shape))
    
    # Broadcast range of times
    subarray = V[t1:t2]
    print('Broadcast Range of Times: V[t1:t2]')
    print('  x0={0}, subarray={1}'.format(subarray.x0.shape, subarray.shape))
    
    # Scalar time
    subarray = V[t1, :]
    print('Scalar Time: V[t1, :]')
    print('  x0={0}, subarray={1}'.format(subarray.x0.shape, subarray.shape))
    
    # List of times
    subarray = V[[t1, t2], :]
    print('List of Times: V[[t1, t2], :]')
    print('  x0={0}, subarray={1}'.format(subarray.x0.shape, subarray.shape))
    
    # Range of times
    subarray = V[t1:t2, :]
    print('Range of Times: V[t1:t2, :]')
    print('  x0={0}, subarray={1}'.format(subarray.x0.shape, subarray.shape))


def test_matrix_get_item():
    
    # TODO
    #   Test Ellipses
    
    # Time interval
    t1 = np.datetime64('2015-12-06T00:00:00')
    t2 = np.datetime64('2015-12-06T10:09:33')
    dt = np.timedelta64(5, 's')
    t = np.arange(t1, t2, dt, dtype='datetime64[ms]')
    npts = len(t)
    
    # Scalar, Vector, and Multi-dimensional time series data
    V = MetaArray(np.random.rand(npts, 32, 32), t, name='velocity', cache=False)
    V.x1 = np.arange(0, 360, 11.25)
    V.x2 = np.logspace(np.log10(10), np.log10(22000), 32)
    
    # Sub-interval for indexing
    t1 = np.datetime64('2015-12-06T00:23:04')
    t2 = np.datetime64('2015-12-06T00:25:34')
    
    print('V.shape = {0}, type(V) = {1}'.format(V.shape, type(V)))
    
    # Broadcast scalar index
    subarray = V[0]
    print('Broadcast Scalar Index: V[0]')
    print('  subarray={0}, x0={1}, x1={2}, x2={3}'.format(subarray.shape, subarray.x0.shape, subarray.x1.shape, subarray.x2.shape))
    
    # Broadcast list of indices
    subarray = V[[0,10,20,30]]
    print('Broadcast List of Indices: V[[...]]')
    print('  subarray={0}, x0={1}, x1={2}, x2={3}'.format(subarray.shape, subarray.x0.shape, subarray.x1.shape, subarray.x2.shape))
    
    # Broadcast index range
    subarray = V[0:50:10]
    print('Broadcast Index Range: V[start:stop:step]')
    print('  subarray={0}, x0={1}, x1={2}, x2={3}'.format(subarray.shape, subarray.x0.shape, subarray.x1.shape, subarray.x2.shape))
    
    # Scalar index
    subarray = V[0, :, :]
    print('Scalar Index: V[0, :, :]')
    print('  subarray={0}, x0={1}, x1={2}, x2={3}'.format(subarray.shape, subarray.x0.shape, subarray.x1.shape, subarray.x2.shape))
    
    # List of indices
    subarray = V[[0,10,20,30], :, :]
    print('List of Indices: V[[...], :, :]')
    print('  subarray={0}, x0={1}, x1={2}, x2={3}'.format(subarray.shape, subarray.x0.shape, subarray.x1.shape, subarray.x2.shape))
    
    # Range of indices
    subarray = V[0:50:10, :, :]
    print('Range of Indices: V[start:stop:step, :, :]')
    print('  subarray={0}, x0={1}, x1={2}, x2={3}'.format(subarray.shape, subarray.x0.shape, subarray.x1.shape, subarray.x2.shape))
    
    # Broadcast scalar time
    subarray = V[t1]
    print('Broadcast Scalar Time: V[t1]')
    print('  subarray={0}, x0={1}, x1={2}, x2={3}'.format(subarray.shape, subarray.x0.shape, subarray.x1.shape, subarray.x2.shape))
    
    # Broadcast list of times
    subarray = V[[t1, t2]]
    print('Broadcast List of Times: V[[t1,t2]]')
    print('  subarray={0}, x0={1}, x1={2}, x2={3}'.format(subarray.shape, subarray.x0.shape, subarray.x1.shape, subarray.x2.shape))
    
    # Broadcast range of times
    subarray = V[t1:t2]
    print('Broadcast Range of Times: V[t1:t2]')
    print('  subarray={0}, x0={1}, x1={2}, x2={3}'.format(subarray.shape, subarray.x0.shape, subarray.x1.shape, subarray.x2.shape))
    
    # Scalar time
    subarray = V[t1, :, :]
    print('Scalar Time: V[t1, :, :]')
    print('  subarray={0}, x0={1}, x1={2}, x2={3}'.format(subarray.shape, subarray.x0.shape, subarray.x1.shape, subarray.x2.shape))
    
    # List of times
    subarray = V[[t1, t2], :, :]
    print('List of Times: V[[t1, t2], :, :]')
    print('  subarray={0}, x0={1}, x1={2}, x2={3}'.format(subarray.shape, subarray.x0.shape, subarray.x1.shape, subarray.x2.shape))
    
    # Range of times
    subarray = V[t1:t2, :, :]
    print('Range of Times: V[t1:t2, :, :]')
    print('  subarray={0}, x0={1}, x1={2}, x2={3}'.format(subarray.shape, subarray.x0.shape, subarray.x1.shape, subarray.x2.shape))
    
    
def test_mms_fpi():
    from pymms.pymms import mrmms_sdc_api as api
    
    sc = 'mms1'
    instr = 'fpi'
    mode = 'fast'
    level = 'l2'
    t0 = np.datetime64('2016-10-22T00:00:00')
    t1 = np.datetime64('2016-10-22T10:09:33')
    
    # Get the data file
    sdc = api.MrMMS_SDC_API(sc, instr, mode, level, 
                            optdesc='des-moms', 
                            start_date='2016-10-22', 
                            end_date='2016-10-23')
    sdc.offline = False
    file = sdc.Download()
    
    # Variable name
    n_vname = '_'.join((sc, 'des', 'numberdensity', mode))
    v_vname = '_'.join((sc, 'des', 'bulkv', 'gse', mode))
    espec_vname = '_'.join((sc, 'des', 'energyspectr', 'omni', mode))
    
    # Read data
    N = pyarray.metabase.from_cdf(file, n_vname, cache=True, clobber=True)
    V = pyarray.metabase.from_cdf(file, v_vname, cache=True, clobber=True)
    ESpec = pyarray.metabase.from_cdf(file, espec_vname, cache=True, clobber=True)
    
    # Plot data
    pyarray.MetaCache.plot()
    
    N_short = N[t0:t1]
    N_short.name = n_vname + '_short'
    N_short.cache()
    
    V_short = V[t0:t1]
    V_short.name = v_vname + '_short'
    V_short.cache()
    
    ESpec_short = ESpec[t0:t1]
    ESpec_short.name = espec_vname + '_short'
    ESpec.cache()
    
    pyarray.MetaCache.plot([N_short, V_short, ESpec_short])
    
    return file
    
    
def main_test():
    data = np.arange(0,10,1)
    time = np.datetime64('now') + np.timedelta64(1, 'm')*np.arange(0,10,1)
    ma = MetaArray(data)
    ma.x0 = time
    ma.plot()


if __name__ == '__main__':
    main_mms_fpi()

