# Instructions for subclassing numpy.ndarray are taken from (9/8/2017)
#   https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
#
# Logic I had in mind:
#   - https://stackoverflow.com/questions/12101958/how-to-keep-track-of-class-instances
#   - https://softwareengineering.stackexchange.com/questions/344859/is-it-good-practice-to-store-instances-within-a-class-variable-in-python
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as img
from matplotlib import dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import datetime as dt
import pdb
import pytz
import re
#from pyarray.metaarray import metaarray, metatime

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
        if key in cls._cache:
            rex = re.compile(key + '(_(\d+))?$')
            num = [re.match(rex, x).group(2) for x in cls._cache if re.match(rex, x)]
            
            if num[0] is None:
                num = 1
            else:
                num = max(num) + 1
            
            new_key = key + '_' + str(num)
        else:
            new_key = key
        
        return new_key
    
    
    @classmethod
    def iscached(cls, names):
        return [name for name in names if name in cls._cache]
    
    
    @classmethod
    def names(cls):
        return list(cls._cache.keys())
    
    
    @classmethod
    def plot(cls, variables=None, nrows=None, ncols=None,
             xlim=None):
        if variables is None:
            variables = cls._cache.keys()
        if isinstance(variables, str):
            variables = [variables]
        
        # Get the object references
        vars = [cls._cache[var]
                if isinstance(var, str)
                else var
                for var in variables
                ]

        # Plot layout
        if (nrows is None) & (ncols is None):
            nrows = len(vars)
            ncols = 1

        # Setup plot
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False)
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        
        if xlim is None:
            xlim = np.asarray(['2100-01-01T00:00:00',
                               '1900-01-01T00:00:00'],
                              dtype='datetime64[s]'
                              )

        # Plot each variable
        for idx, var in enumerate(vars):
            # 2D data are plotted as images
            if hasattr(var, 'x1'):
                var.image(axes=axes[idx,0], show=False)
            # 1D data are plotted as scatter plots
            elif hasattr(var, 'x0'):
                var.plot(axes=axes[idx,0], show=False)
            
            # Find a common x-range
            xlim[0] = min(xlim[0], var.x0[0])
            xlim[1] = max(xlim[1], var.x0[-1])
            
            # Do the following:
            #   - For all but the bottom row, remove all labels and
            #     annotations from the x-axis
            #   - For all but the top row, remove all titles
            ax = axes[idx,0]
            if idx < nrows - 1:
                ax.set_xticks([])
                ax.set_xlabel('')
            if idx > 0:
                ax.set_title('')
            
            # On the last row, format the x-axis ticks as dates
            if idx == nrows - 1:
                axes[idx,0].xaxis.set_major_locator(locator)
                axes[idx,0].xaxis.set_major_formatter(formatter)

        # Make pretty by
        #   - Increasing the right-margin to make room for
        #     colorbars and legends
        #   - Make the X-axis limits the same for all plots
        plt.setp(axes, xlim=xlim)
        plt.subplots_adjust(right=0.85)
        return fig, axes


class MetaBase(np.ndarray):
    
    def __new__(cls, x, cache=False, dtype=None, name='MetaBase', **kwargs):
        # Convert input to ndarray (if it is not one already)
        # Cast it as an instance of MrArray
        obj = np.asarray(x, dtype=dtype).view(cls)
        
        # Name and cache the object
        obj.name = name
        if cache:
            MetaBase.cache(obj)
        
        # Add new attributes
        for key, value in kwargs.items():
            setattr(obj, key, value)
        
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
        self.name = getattr(obj, 'name', 'metaarray')
        # We do not need to return anything
    
    def __getitem__(self, idx):
        obj = super(MetaBase, self).__getitem__(idx)
        
        # Scalars do not have attributes (i.e. x0, x1, x2, etc.)
        if np.isscalar(obj):
            return obj
        
        # Copy attributes to the new object, but not the
        # dependent variables because those will be updated
        # below.
        attr_names = [key
                      for key in  self.__dict__
                      if (key[0] != 'x') or (not key[1:].isdigit())
                      ]
        self.copy_attr_to(obj, attr_names)
        
        # Get the proper sub-array of dependent variable data.
        # Make sure that any shared data is shared to save
        # memory.
        
        # integer index   ==> idx is an int
        # List of indices ==> idx is a list
        # start:stop:step ==> idx is a slice
        # multi-dim index ==> idx is a tuple
        
        # Organize the dependent variable attributes
        indices = self._expand_index(idx)
        self._depend_assigner(obj, indices)
        #if isinstance(indices, (list, slice)):
        #    return obj
        #self._depend_arrange(obj, indices)
        
        return obj
    
    def cache(self, **kwargs):
        MetaCache.cache(self, **kwargs)
    
    
    def copy_attr_to(self, dest, attr_names=None):
        '''
        Copy attributes to another object.
        
        Parameters
        ----------
        dest : `MetaBase` or subclass
            Instance object to which attributes are copied. If an attribute
            of `self` is already in `dest`, that attribute is skipped.
        attr_names : str or list of str
            Names of the attributes to be copied. By default, all attributes
            in `__dict__` are copied.
        '''
        if attr_names is None:
            attr_names = self.__dict__.keys()
        
        for key in attr_names:
            if key not in dest.__dict__:
                setattr(dest, key, getattr(self, key))
    
#     def __del__(self):
#         try:
#             self.remove()
#         except ValueError:
#             pass
    
    def image(self, axes=None, colorbar=True, show=False):
        # Create the figure
        if axes is None:
            fig, axes = plt.subplots(nrows=1, ncols=1)
        
        # Convert time to seconds and reshape to 2D arrays
        x0 = mdates.date2num(self.x0)
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
        
        # Create a colorbar to the right of the image
        if colorbar:
            cbaxes = inset_axes(axes,
                                width='1%', height='100%', loc=4,
                                bbox_to_anchor=(0, 0, 1.05, 1),
                                bbox_transform=axes.transAxes,
                                borderpad=0)
            cb = plt.colorbar(im, cax=cbaxes, orientation='vertical')
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
    
    
    def plot(self, axes=[], legend=True, show=False):
        if not axes:
            axes = plt.axes()
        
        # Plot the data
        lines = axes.plot(mdates.date2num(self.x0), self)
        #axes.figure.autofmt_xdate()
        
        # Set plot attributes
        self._plot_apply_xattrs(axes, self.x0)
        self._plot_apply_yattrs(axes, self)
        
        if legend:
            try:
                # Set the label for each line so that they can
                # be returned by Legend.get_legend_handles_labels()
                for line, label in zip(lines, self.label):
                    line.set_label(label)
                
                # Create the legend outside the right-most axes
                leg = axes.legend(bbox_to_anchor=(1.05, 1),
                                  borderaxespad=0.0,
                                  frameon=False,
                                  handlelength=0,
                                  handletextpad=0,
                                  loc='upper left')
                
                # Color the text the same as the lines
                for line, text in zip(lines, leg.get_texts()):
                    text.set_color(line.get_color())
                
            except AttributeError:
                pass
    
    def remove(self):
        try:
            del MetaCache[self]
        except ValueError:
            pass
    
    def _depend_arrange(self, obj, indices):
        '''
        Adjust the axis to which the dependent variables are associated.
        An index of None type will add a dimension while an integer index
        will remove a dimension.
        
        Parameters
        ----------
        obj : `type(self)`
            The subarray of self for which the dependent variable attributes
            need to be rearranged
        indices : tuple
            The expanded set of indices (`_expand_index()`) that created `obj`
        
        Returns
        -------
        attr_nums : `np.ndarray`
            Adjusted dependent attribute numbers
        attr_mask : list of bool
            `True` if the corresponding element of `attr_nums` refers to
            a dimension that will exist in `obj` after `indices` are
            applied, `False` otherwise. A dimension will be removed if
            the index along that dimension is an integer.
        '''
        
        # Determine how the old attributes map to the new attributes
        depnum = lambda x: int(x[1:])
        dep_count = 0
        attr_nums = np.arange(self.ndim)
        attr_mask = [True] * self.ndim
        
        # If indices reference a single dimension only,
        # there is no need to re-arrange the dependent variables
        if isinstance(indices, (list, slice)):
            return attr_nums, attr_mask
        
        for idx in indices:
            # If an index is an integer, a shallow dimension is created that
            # is automatically squeezed out of the array. All dependent
            # variables beyond that dimension must be shifted down a dimension
            if isinstance(idx, int):
                attr_mask[dep_count] = False
                attr_nums[dep_count+1:] -= 1
                dep_count += 1
            
            # If an indices is None, a new dimension is created and all
            # dependent variables at and beyond it must be shifted up a
            # dimension
            elif idx == None:
                attr_nums[dep_count:] += 1
            
            else:
                dep_count += 1
        
        return attr_nums, attr_mask
        
    
    def _depend_assigner(self, obj, indices):
        '''
        For the subarray of `self` and the indices that created it, also
        extract the corresponding subarray of each dependent variable and
        determine which have been squeezed out by automatic removal of
        shallow dimensions
        
        Parameters
        ----------
        obj : `type(self)`
            The subarray of `self` created by `__getitem__` to which
            dependent variables are to be assigned.
        indices : tuple
            The expanded set of indices (`_expand_index()`) that created `obj`
        '''
        # Get relevant items from the dependent variables
        dep_attr_nums = np.arange(self.ndim)
        new_attr_nums = self._depend_arrange(obj, indices)[0]
        for attr_num in dep_attr_nums:
            attr_name = 'x{0:d}'.format(attr_num)
            
            try:
                attr_value = getattr(self, attr_name)
            # Attribute errors can occurs if self does not have
            # the dependent variable
            except AttributeError:
                continue
            
            attr_value = self._depend_getitem(attr_value,
                                              indices, 
                                              attr_num
                                              )
            # If dep_value is None, that dimension has been
            # squeezed out of the array and the corresponding
            # dependent variable can be ignored.
            if attr_value is not None:
                new_attr_name = 'x{0:d}'.format(new_attr_nums[attr_num])
                setattr(obj, new_attr_name, attr_value)
    
    def _depend_getitem(self, depend, indices, axis):
        '''
        Retrieve a sub-array from a dependent variable.
        
        Parameters
        ----------
        depend : `numpy.ndarray` or subclass
            The dependent variable for which a sub-array is to be returned
        indices : tuple
            The expanded set of indices (`_expand_index()`) to access `depend`
        axis : int
            Axis of self for which `depend` is the dependent variable
        
        Returns
        -------
        subarray : `numpy.ndarray` or subclass
            The sub-array of `depend`
        '''
        try:
            return depend[indices]
        
        # Index error if idx is a tuple (multi-dimensional indexing)
        #   - Select the indices along the dependent axis
        #   - A scalar index will create an empty dimension, which
        #     numpy squeezes out of the array. In this case, the
        #     dependent variable is no longer relevant and can be
        #     ignored.
        except IndexError:
            if isinstance(indices, tuple) and (len(indices) == self.ndim):
                if isinstance(indices[axis], int):
                    return None
                elif depend.ndim == 2:
                    return depend[indices[0], indices[axis]]
                elif depend.ndim == 1:
                    return depend[indices[axis]]
                else:
                    raise IndexError('unable to apply indices ' \
                                     'to dependent variable')
            else:
                IndexError('too many indices for array')
    
    def _expand_index(self, idx):
        '''
        Expand a set of indices into a tuple of multi-dimensional
        indices. Any dimensions that are broadcast are explicitly
        assigned Slice(None, None, None).
        
        Parameters
        ----------
        idx : int, list, tuple
            Indices intended to be used to extract a subarray of `self`
        
        Returns
        -------
        indices : list, tuple
            Expanded set of indices
        '''
        # If idx indexes fewer axes than self has, expand
        # idx to reference all dimensions. This is equivalent
        # to expanding an Ellipsis and the end of the slice
        # [0,Ellipsis]
        indices = idx
        if isinstance(idx, int) and (self.ndim > 1):
            indices = (idx, *[slice(None, None, None) 
                              for i in range(self.ndim - 1)])
        
        elif isinstance(idx, tuple) and (len(idx) < self.ndim):
            indices = (*idx, *[slice(None, None, None) 
                               for i in range(self.ndim - len(idx))])
        
        return indices
    
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



