# Instructions for subclassing numpy.ndarray are taken from (9/8/2017)
#   https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
import numpy as np
from pyarray.metaarray import metaarray
import pdb

class MetaDepend(metaarray.MetaArray):
    
    def __setattr__(self, name, value):
        if (name[0] == 'x') and (name[1:].isdigit()):
            axis = int(name[1:])
            if axis != 0:
                raise AttributeError('x0 is the only allowed ' \
                                     'dependent variable attribute'
                                     )

        # Set the value
        super(MetaDepend, self).__setattr__(name, value)

#     def _depend_getitem(self, indices, axis, parent):
#         '''
#         Retrieve a sub-array based on the indices into a parent variable.
#         
#         Parameters
#         ----------
#         indices : tuple
#             The expanded set of indices (`_expand_index()`) to access `depend`
#         axis : int
#             Axis of self for which `depend` is the dependent variable
#         depend : `numpy.ndarray` or subclass
#             The parent object for which `self` is a dependent variable
#         
#         Returns
#         -------
#         subarray : `numpy.ndarray` or subclass
#             The sub-array
#         '''
#         try:
#             return self[indices]
#         
#         # Index error if idx is a tuple (multi-dimensional indexing)
#         #   - Select the indices along the dependent axis
#         #   - A scalar index will create an empty dimension, which
#         #     numpy squeezes out of the array. In this case, the
#         #     dependent variable is no longer relevant and can be
#         #     ignored.
#         except IndexError:
#             if isinstance(indices, tuple) and (len(indices) == parent.ndim):
#                 if self.ndim == 2:
#                     pdb.set_trace()
#                     return self[indices[0], indices[axis]]
#                 elif self.ndim == 1:
#                     return self[indices[axis]]
#                 elif isinstance(indices[axis], int):
#                     return None
#                 else:
#                     raise IndexError('unable to apply parent index ' \
#                                      'to dependent variable')
#             else:
#                 IndexError('too many indices for array')
