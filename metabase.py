# Instructions for subclassing numpy.ndarray are taken from (9/8/2017)
#   https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
#
# Logic I had in mind:
#   - https://stackoverflow.com/questions/12101958/how-to-keep-track-of-class-instances
#   - https://softwareengineering.stackexchange.com/questions/344859/is-it-good-practice-to-store-instances-within-a-class-variable-in-python
import numpy as np
from spacepy import pycdf
from matplotlib import pyplot as plt
from matplotlib import image as img
import datetime as dt
import pyarray
import pdb
import pytz
import re

"""
Generate self-documented data and time series objects. Allow objects to be cached
for future analysis and exploration. Provide an automated mechanism to visualize cached
or local objects. Import data from CDF files.
"""


class MetaCache():
    _cache = dict()
    
    @classmethod
    def cache(cls, objects, clobber=False):
        if not isinstance(objects, list):
            objects = [objects]
        
        # Check if the name already exists. When clobbering,
        # delete the existing value. Otherwise, if the name
        # exists, find the highest number in <name>_# that
        # makes <name> unique.
        for obj in objects:
            if obj.name in cls._cache:
                if clobber:
                    name = obj.name
                else:
                    name = cls.get_unique_key(obj.name)
                    obj.name = name
            else:
                name = obj.name
        
            # Add to the variable cache
            cls._cache[name] = obj
    
    
    @classmethod
    def get_unique_key(cls, key):
        rex = re.compile(key + '(_(\d+))?$')
        num = [re.match(rex, x).group(2) for x in cls._cache if re.match(rex, x)]
        
        try:
            num = max(num)
        except ValueError:
            num = 0
        
        return key + '_' + str(num+1)
    
    
    @classmethod
    def iscached(cls, names):
        return [name for name in names if name in cls._cache]
    
    
    @classmethod
    def names(cls):
        return list(cls._cache.keys())
    
    
    @classmethod
    def plot(cls, variables=None, nrows=None, ncols=None):
        if not variables:
            variables = cls._cache.keys()
        if isinstance(variables, str):
            variables = [variables]
        
        # Get the object references
        vars = [cls._cache[var] for var in variables]

        # Plot layout
        if not nrows and not ncols:
            nrows = len(vars)
            ncols = 1

        # Setup plot
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

        # Plot each variable
        for idx, var in enumerate(vars):
            if hasattr(var, 'x1'):
                var.image(axes=axes[idx], show=False)
            elif hasattr(var, 'x0'):
                var.plot(axes=axes[idx], show=False)

        # Display the figure
        plt.show()


class MetaBase(np.ndarray):
    
    def __new__(cls, x, cache=False, dtype=None, name='MetaBase'):
        # Convert input to ndarray (if it is not one already)
        # Cast it as an instance of MrArray
        obj = np.asarray(x, dtype=dtype).view(cls)
        
        # Name and cache the object
        obj.name = name
        if cache:
            MetaBase.cache(obj)
        
        # Finally, we must return the newly created object:
        return obj

    
    def __array_finalize__(self, obj):
        # ``self`` is a new object resulting from
        # ndarray.__new__(InfoArray, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. InfoArray():
        #    obj is None
        #    (we're in the middle of the InfoArray.__new__
        #    constructor, and self.info will be set when we return to
        #    InfoArray.__new__)
        
        if obj is None: return
        # From view casting - e.g arr.view(InfoArray):
        #    obj is arr
        #    (type(obj) can be InfoArray)
        # From new-from-template - e.g infoarr[:3]
        #    type(obj) is InfoArray
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'info', because this
        # method sees all creation of default objects - with the
        # InfoArray.__new__ constructor, but also with
        # arr.view(InfoArray).
        self.name = getattr(obj, 'name', 'MetaBase')
        # We do not need to return anything
    
    
    def cache(self, **kwargs):
        MetaCache.cache(self, **kwargs)
    
    
#     def __del__(self):
#         try:
#             self.remove()
#         except ValueError:
#             pass
    
    
    def image(self, axes=[], colorbar=True, show=True):
        # Create the figure
        if not axes:
            fig, axes = plt.subplots(nrows=1, ncols=1)
        
        # Convert time to seconds and reshape to 2D arrays
        x0 = (self.x0 - self.x0[0]).astype('timedelta64[s]')
        x0 = np.array([t.item().total_seconds() for t in x0])
        x1 = self.x1
        if x0.ndim == 1:
            x0 = np.repeat(x0[:, np.newaxis], self.shape[1], axis=1)
        if x1.ndim == 1:
            x1 = np.repeat(x1[np.newaxis, :], self.shape[0], axis=0)
        
        # Format the image
        #   TODO: Use the delta_plus and delta_minus attributes to create
        #         x0 and x1 arrays so that x does not lose an element
        data = self[0:-1,0:-1]
        if hasattr(self, 'scale'):
            data = np.ma.log(data)
        
        # Create the image
        im = axes.pcolorfast(x0, x1, data, cmap='nipy_spectral')
        axes.images.append(im)
        
        # Set plot attributes
        self._plot_apply_xattrs(axes, self.x0)
        self._plot_apply_yattrs(axes, x1)
        
        # TODO: Add a colorbar
        if colorbar:
            cb = plt.colorbar(im)
            try:
                cb.set_label(self.title)
            except AttributeError:
                pass
        
        # Display the plot
        if show:
            plt.ion()
            plt.show()
    
    
    def iscached(self):
        # Determine if the instance has been cached
        return self.name in MetaCache._cache
    
    
    def plot(self, axes=[], legend=True, show=True):
        if not axes:
            axes = plt.axes()
        
        # Plot the data
        axes.plot(self.x0.astype(dt.datetime), self)
        
        # Set plot attributes
        self._plot_apply_xattrs(axes, self.x0)
        self._plot_apply_yattrs(axes, self)
        
        if legend:
            try:
                axes.legend(self.label)
            except AttributeError:
                pass
        
        # Display the plot
        if show:
            plt.ion()
            plt.show()
    
    
    def remove(self):
        try:
            del MetaCache[self]
        except ValueError:
            pass
    
    
    @staticmethod
    def _plot_apply_xattrs(ax, x):
        try:
            ax.XLim = x.lim
        except AttributeError:
            pass
        
        try:
            ax.set_title(x.plot_title)
        except AttributeError:
            pass
        
        try:
            ax.set_xscale(x.scale)
        except AttributeError:
            pass
        
        try:
            ax.set_xlabel(x.title.replace('\n', ' '))
        except AttributeError:
            pass
        
    
    @staticmethod
    def _plot_apply_yattrs(ax, y):
        try:
            ax.YLim = y.lim
        except AttributeError:
            pass
        
        try:
            ax.set_title(y.plot_title)
        except AttributeError:
            pass
        
        try:
            ax.set_yscale(y.scale)
        except AttributeError:
            pass
        
        try:
            ax.set_ylabel(y.title)
        except AttributeError:
            pass

    
#     @classmethod
#     def get(cls, arrlist=[]):
#         # Return all variables
#         if not arrlist:
#             return cls._cache
#         
#         # Output array
#         arrs = []
#         
#         # Get an array from the cache
#         for item in arrlist:
#             if isinstance(item, int):
#                 arrs.append(cls._cache[item])
#             
#             elif isinstance(item, str):
#                 names = cls.names()
#                 arrs.append(cls._cache[names.index(item)])
#             
#             elif item is MetaBase:
#                 arrs.apend(item)
#             
#             else:
#                 raise TypeError('arrlist must contain integers, strings, or MrArray objects.')
#         
#         # Return a single object if only one input was given
#         if not isinstance(item, list):
#             arrs = arrs[0]
#         
#         return arrs

    
#     @classmethod
#     def names(cls):
#         names = [arr.name for arr in cls._cache]
#         return names
#         
#         # Print the index and name of each item in the cache.
#         # Use a space to pad between index and name; index is
#         # right-justified and padded on the left while name
#         # would be left-justified and padded on the right. How
#         # then to pad between them?
#         if len(cls._cache) == 0:
#             print('The cache is empty.')
#         else:
#             print('{:4}{:3}{}'.format('Index', '', 'Name'))
#             for idx, item in enumerate(cls._cache):
#                 print('{0:>4d}{1:<4}{2}'.format(idx, '', item.name))


def from_cdf(files, variable, cache=False, clobber=False, name=None):
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
            var = _from_cdf_read_var(f, variable)
    
    # Cache
    if cache:
        var.cache(clobber=clobber)
    
    return var


def _from_cdf_read_var(cdf, varname):
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
            var = pyarray.MetaTime(cdf_var[...])
        else:
            var = pyarray.MetaArray(cdf_var[...])
    
        # Append to existing data
        #   TODO: Add ufunc to MetaBase to np.append returns MetaBase sub/class
        if varname in cdf_vars and cdf_var.rv():
            d0 = cdf_vars[varname]
            var = np.append(var, d0, 0).view(type(var))
        
        var.name = varname
        var.rec_vary = cdf_var.rv()
    
        # Mark as read
        #  - Prevent infinite loop. Must save the variable in the registry
        #  so that variable attributes do not try to read the same variable
        #  again.
        cdf_vars[varname] = var
        file_vars[varname] = var
    
        # Read the metadata
        var = _from_cdf_read_var_attrs(cdf, var)
        var = _from_cdf_attrs2gfxkeywords(cdf, var)

    return var


def _from_cdf_read_var_attrs(cdf, var):
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
    var.cdf_name = cdf_var.name()
    var.cdf_type = cdf_var.type()
    
    # Variable attributes
    for vattr in cdf_var.attrs:
    
        # Follow pointers
        attrvalue = cdf_var.attrs[vattr]
        if isinstance(attrvalue, str) and attrvalue in cdf:
            varname = attrvalue
            attrvalue = _from_cdf_read_var(cdf, varname)
        
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


