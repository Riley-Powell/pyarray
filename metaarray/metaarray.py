# Instructions for subclassing numpy.ndarray are taken from (9/8/2017)
#   https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
import numpy as np
import datetime as dt
from spacepy import pycdf
from cdflib import cdfread
from pyarray.metaarray import metabase, metatime
import pdb

class MetaArray(metabase.MetaBase):
    
    def __getitem__(self, idx):
        try:
            return super(MetaArray, self).__getitem__(idx)
        
        # IndexError raised 
        #    - when idx is a datetime64 object
        #    - when idx is out of bounds integer
        #    - when idx is multi-dimensional index (tuple) with too many elements
        # TypeError raised when slice contains datetime64 object
        except (TypeError, IndexError):
            if not self._index_has_time(idx):
                raise
            
        # If idx is a datetime64, get the associated index. An
        # IndexError will occur if idx is a tuple of multi-
        # dimensional indices.
        try:
            index = self.x0.get_item_index(idx)
            return self[index]
        except IndexError:
            pass
        
        
        # If idx is a tuple, the first element should access the
        # time dimension. If the slice contains integer indices
        # instead of datetime64 indices, a 
        index = (self.x0.get_item_index(idx[0]), *idx[1:])
            
        return self[index]
        
#        # TODO: Nearest-neighbor interpolation if dt is a time
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
    
#     def __setattr__(self, name, value):
#         if (name[0] == 'x') and (name[1:].isdigit()):
#             axis = int(name[1:])
#             
#             # Test for a datetime object without importing datetime
#             # by checking if the value has a date method.
#             try:
#                 value.date()
#                 value = metatime.Metatime(value, axis=axis)
#                 super(MetaArray, self).__setattr__(name, value)
#                 return
#             
#             # Attribute error will occur if value does not have a
#             # date method. Need to try something else.
#             except AttributeError:
#                 pass
#                 
#             # Test for a datetime64 array by checking its dtype.
#             # If not datetime64, pass out of try-except block
#             try:
#                 if np.issubdtype(value.dtype, np.datetime64):
#                     value = metatime.MetaTime(value)
#                     super(MetaArray, self).__setattr__(name, value)
#                     return
#             
#             # If no dtype attribute is present, then value is not
#             # an ndarray.
#             except AttributeError:
#                 pass
#             
#             # If the variable is not a time-type, treat it as a
#             # regular dependent variable. Do not create a new
#             # instance if not required so that the attributes are
#             # not lost
#             if not isinstance(value, (metadepend.MetaDepend,
#                                       metatime.MetaTime)
#                               ):
#                 value = metadepend.MetaDepend(value)
# 
#         # Set the value
#         super(MetaArray, self).__setattr__(name, value)
    
    def _check_time(self, t):
        return np.all(self.x0 == t)
    
    def _index_has_time():
        def check_index(idx):
            # Success if it is a time-type 
            if (isinstance(idx, (np.datetime64, dt.datetime)) or
                (isinstance(idx, np.ndarray) and
                 idx.dtype.name.startswith('datetime64')
                 )
                ):
                return True
            
            # All values of the list or slice must be time-like
            # because all values from __getindex__ are then
            # passed to self.x0 to look up the indices of those
            # time values
            elif isinstance(idx, list):
                return all(check_index(i) for i in idx)
            elif isinstance(idx, slice):
                return all(check_index(i)
                           for i in (idx.start, idx.stop))
            
            # Tuples contain multi-dimensional indices. Time
            # is expected only in the first dimension, so check
            # the first element
            elif isinstance(idx, tuple):
                return check_index(idx[0])
            
            # Non-container types are False (e.g. int)
            else:
                print(type(idx))
                return False
            
        return check_index
    _index_has_time = staticmethod(_index_has_time())


def from_cdflib(files, variable, cache=False, clobber=False, name=None):
    """
    """
    global cdf_vars
    global file_vars

    if isinstance(files, str):
        files = [files]
    
    # Read variables from file
    cdf_vars = {}
    for file in files:
        file_vars = {}
        with cdfreader.CDF(file) as f:
            var = _from_cdflib_read_var(f, variable)
    
    if cache:
        var.cache(clobber=clobber)
    
    return var


def _from_cdflib_read_var(cdf, variable):
    """
    """
    global cdf_vars
    global file_vars
    epoch_types = [31, 32, 33]  # ['CDF_EPOCH', 'CDF_EPOCH16', 'CDF_TIME_TT2000']
    
    # Data has already been read from this CDF file
    varinq = cdf.varinq(varname)
    if varname in file_vars:
        var = file_vars[varname]
    else:
        if varinq['Data_Type'] in epoch_types:
            var = metatime.MetaTime(cdf.varget(varname))
        else:
            var = metaarray.MetaArray(cdf.varget(varname))
    
        # TODO: Append to existing data
        #   Add ufunc to MetaBase to np.append returns MetaBase sub/class
        if varname in cdf_vars and varinq['Rec_Var']:
            d0 = cdf_vars[varname]
            var = np.append(d0, var, 0).view(type(var))
        
        var.name = varname
        var.rec_vary = varinq['Rec_Var']
        var.cdf_name = varname
        var.cdf_type = varinq['Data_Type']
    
        # List as read
        #  - Prevent infinite loop. Must save the variable in the registry
        #  so that variable attributes do not try to read the same variable
        #  again.
        cdf_vars[varname] = var
        file_vars[varname] = var
    
        # Read the metadata
        var = _from_cdflib_read_var_attrs(cdf, var)
        var = _from_cdf_attrs2gfxkeywords(cdf, var)
    
    return var


def _from_cdflib_read_var_attrs(cdf, var):
    """
    """
    vattrs = cdf.varattsget(var.name)
    for attrname in vattrs:
        # Follow pointers
        if isinstance(vattrs[attrname], str) and \
                vattrs[attrname] in cdf:
            varname = vattrs[attrname]
            attrvalue = _from_cdflib_read_var(cdf, varname)
        else:
            attrvalue = vattrs.pop(attrname)
        
        # Set the attribute value
        if attrname == 'DELTA_PLUS_VAR':
            var.delta_plus = attrvalue
        elif attrname == 'DELTA_MINUS_VAR':
            var.delta_minus = attrvalue
        elif attrname == 'DEPEND_0':
            var.x0 = attrvalue
        elif attrname == 'DEPEND_1':
            var.x1 = attrvalue
        elif attrname == 'DEPEND_2':
            var.x2 = attrvalue
        elif attrname == 'DEPEND_3':
            var.x3 = attrvalue
#        elif attrname == 'LABL_PTR_1':
#            var.label = attrvalue
        elif attrname == 'LABL_PTR_2':
            var.label2 = attrvalue
        elif attrname == 'LABL_PTR_3':
            var.label3 = attrvalue
        else:
            setattr(var, attrname, attrvalue)
    
    return var



def from_pycdf(files, variable,
               cache=False, clobber=False, name=None,
               tstart=None, tend=None):
    """
    Read variable data from a CDF file.

    Parameters
    ==========
    filenames : str, list
        Name of the CDF file(s) to be read.
    variable : str
        Name of the variable to be read.

    Returns
    =======
    vars : mrarry
        A mrarray object.
    """
    global cdf_vars
    global file_vars
    
    if isinstance(files, str):
        files = [files]

    # Read variables from files
    cdf_vars = {}
    for file in files:
        file_vars = {}
        with pycdf.CDF(file) as f:
            var = _from_pycdf_read_var(f, variable)
    
    # Trim time range
    if (tstart is not None) or (tend is not None):
        if tstart is not None:
            istart = var.x0[0]
        if tend is not None:
            iend = var.x0[-1]
        var = var[tstart:tend]
    
    # Cache
    if cache:
        var.cache(clobber=clobber)
    
    
    return var


def _from_pycdf_read_var(cdf, varname):
    """
    Read data and collect metadata from a CDF variable.

    Parameters
    ==========
    cdf: : object
        A spacepy.pycdf.CDF object of the CDF file being read.
    varname : str
        The name of the CDF variable to be read
    """
    global cdf_vars
    global file_vars

    # Data has already been read from this CDF file
    if varname in file_vars:
        var = file_vars[varname]

    else:
        time_types = [pycdf.const.CDF_EPOCH.value,
                      pycdf.const.CDF_EPOCH16.value,
                      pycdf.const.CDF_TIME_TT2000.value]
        
        # Read the data
        cdf_var = cdf[varname]
        if cdf_var.type() in time_types:
            var = metatime.MetaTime(cdf_var[...])
        else:
            var = MetaArray(cdf_var[...])
    
        # TODO: Append to existing data
        #   Add ufunc to MetaBase so np.append returns MetaBase sub/class
        if varname in cdf_vars and cdf_var.rv():
            d0 = cdf_vars[varname]
            var = np.append(d0, var, 0).view(type(var))
        
        var.name = varname
        var.rec_vary = cdf_var.rv()
        var.cdf_name = cdf_var.name()
        var.cdf_type = cdf_var.type()
    
        # List as read
        #  - Prevent infinite loop. Must save the variable in the registry
        #  so that variable attributes do not try to read the same variable
        #  again.
        cdf_vars[varname] = var
        file_vars[varname] = var
    
        # Read the metadata
        var = _from_pycdf_read_var_attrs(cdf, var)
        var = _from_cdf_attrs2gfxkeywords(cdf, var)

    return var


def _from_pycdf_read_var_attrs(cdf, var):
    """
    Read metadata from a CDF variable.
    
    Parameters
    ==========
    cdf: : object
        A spacepy.pycdf.CDF object of the CDF file being read.
    var : object
        The MrArray object that will take on CDF variable attributes
        as object attributes.
    
    Returns
    ==========
    var : object
        The input tsdata object with new attributes.
    """
    
    # CDF variable and properties
    cdf_var = cdf[var.name]
    
    # Variable attributes
    for vattr in cdf_var.attrs:
    
        # Follow pointers
        attrvalue = cdf_var.attrs[vattr]
        if isinstance(attrvalue, str) and attrvalue in cdf:
            varname = attrvalue
            attrvalue = _from_pycdf_read_var(cdf, varname)
        
        # Set the attribute value
        if vattr == 'DELTA_PLUS_VAR':
            var.delta_plus = attrvalue
        elif vattr == 'DELTA_MINUS_VAR':
            var.delta_minus = attrvalue
        elif vattr == 'DEPEND_0':
            var.x0 = attrvalue
        elif vattr == 'DEPEND_1':
            var.x1 = attrvalue
        elif vattr == 'DEPEND_2':
            var.x2 = attrvalue
        elif vattr == 'DEPEND_3':
            var.x3 = attrvalue
#        elif vattr == 'LABL_PTR_1':
#            var.label = attrvalue
        elif vattr == 'LABL_PTR_2':
            var.label2 = attrvalue
        elif vattr == 'LABL_PTR_3':
            var.label3 = attrvalue
        else:
            setattr(var, vattr, attrvalue)
    
    return var


def _from_cdf_attrs2gfxkeywords(cdf, var):
    """
    Set plotting attributes for the variable based on CDF metadata.
    
    Parameters
    ==========
    cdf: : object
        A spacepy.pycdf.CDF object of the CDF file being read.
    var : object
        The tsdata object that will take on CDF variable attributes
        as object attributes.
    
    Returns
    ==========
    var : object
        The input MrArray object with new attributes.
    """
    
    # Extract the attributes for ease of access
    cdf_attrs = cdf[var.name].attrs
    
    # Plot title
    if 'FIELDNAM' in cdf_attrs:
        var.plot_title = cdf_attrs['FIELDNAM']
    elif 'CATDESC' in cdf_attrs:
        var.plot_title = cdf_attrs['CATDESC']
    
    # Axis label
    title = ''
    if 'LABLAXIS' in cdf_attrs:
        title = cdf_attrs['LABLAXIS']
    elif 'FIELDNAM' in cdf_attrs:
        title = cdf_attrs['FIELDNAM']
    if 'UNITS' in cdf_attrs:
        title = title + '\n(' + cdf_attrs['UNITS'] + ')'
    var.title = title
    
    # Legend label
    if 'LABL_PTR_1' in cdf_attrs:
        var.label = cdf[cdf_attrs['LABL_PTR_1']][...]
        
    # Axis scaling
    if 'SCALETYP' in cdf_attrs:
        var.scale = cdf_attrs['SCALETYP']
    
    return var


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
    N = metabase.from_cdf(file, n_vname, cache=True, clobber=True)
    V = metabase.from_cdf(file, v_vname, cache=True, clobber=True)
    ESpec = metabase.from_cdf(file, espec_vname, cache=True, clobber=True)
    
    # Plot data
    metabase.MetaCache.plot()
    
    N_short = N[t0:t1]
    N_short.name = n_vname + '_short'
    N_short.cache()
    
    V_short = V[t0:t1]
    V_short.name = v_vname + '_short'
    V_short.cache()
    
    ESpec_short = ESpec[t0:t1]
    ESpec_short.name = espec_vname + '_short'
    ESpec.cache()
    
    metabase.MetaCache.plot([N_short, V_short, ESpec_short])
    
    return file
    
    
def main_test():
    data = np.arange(0,10,1)
    time = np.datetime64('now') + np.timedelta64(1, 'm')*np.arange(0,10,1)
    ma = MetaArray(data)
    ma.x0 = time
    ma.plot()


if __name__ == '__main__':
    main_mms_fpi()

