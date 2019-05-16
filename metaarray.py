# Instructions for subclassing numpy.ndarray are taken from (9/8/2017)
#   https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
import numpy as np
import pyarray
import pdb

class MetaArray(pyarray.MetaBase):
    
    def __new__(cls, x, t=[], **kwargs):
        
        # Create a numpy array of the data
        obj = super(MetaArray, cls).__new__(cls, x, **kwargs)
        
        # Add new attributes
        obj.t = pyarray.MetaTime(t)
        
        # Finally, we must return the newly created object:
        return obj

    
    def __array_finalize__(self, obj):
        if obj is None: return
        
        super(MetaArray, self).__array_finalize__(obj)
        self.t = getattr(obj, 't', None)
    
    
    def __getitem__(self, idx):
        # use duck typing
        # is it an integer or sequence?
        obj = super(MetaArray, self).__getitem__(idx)
        
        try:
            if isinstance(idx, (int,slice)):
                obj.x0 = self.x0[idx]
            else:
                obj.x0 = self.x0[idx[0]]
        except AttributeError:
            pass
        except:
            raise
        
        return obj
        
        
        # TODO: Nearest-neighbor interpolation if dt is a time
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
    
    
def main_mms_fpi():
    from pymms import pymms
    
    # Get the data file
    sdc = pymms.MrMMS_SDC_API('mms1', 'fpi', 'brst', 'l2', 
                              optdesc='des-moms', 
                              start_date='2016-10-22', 
                              end_date='2016-10-22T10:09:33')
    sdc.offline = False
    file = sdc.Download()
    
    # Variable name
    n_vname = 'mms1_des_numberdensity_brst'
    v_vname = 'mms1_des_bulkv_gse_brst'
    espec_vname = 'mms1_des_energyspectr_omni_brst'
    
    # Read data
    N = pyarray.metabase.from_cdf(file, n_vname, cache=True)
    V = pyarray.metabase.from_cdf(file, v_vname, cache=True)
    ESpec = pyarray.metabase.from_cdf(file, espec_vname, cache=True)
    
    # Plot data
    pyarray.MetaCache.plot()
    
    return file
    
    
def main_test():
    data = np.arange(0,10,1)
    time = np.datetime64('now') + np.timedelta64(1, 'm')*np.arange(0,10,1)
    ma = MetaArray(data)
    ma.x0 = time
    ma.plot()

if __name__ == '__main__':
    main_mms_fpi()

