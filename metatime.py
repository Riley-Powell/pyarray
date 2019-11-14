# Instructions for subclassing numpy.ndarray are taken from (9/8/2017)
#   https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
import numpy as np
import datetime as dt
import pyarray
import re
import pytz
import pdb

class MetaTime(pyarray.MetaBase):
    
    def __new__(cls, x, units='us', **kwargs):
        """
        Create a new instance of the MetaTime class.

        Parameters
        ----------
        cls : obj
            Class type to be created.
        x : list of datetime or datetime64
            Monotonically increasing time array.
        units : str
            Precision of the newly created time array.

        Returns
        -------
        obj : object
            New object instance of class ``cls``.
        """
        
        # Create a datetime64 object
        obj = super(MetaTime, cls).__new__(cls, x, dtype='datetime64['+units+']', **kwargs)
        
        # Finally, we must return the newly created object:
        return obj

    
    def __array_finalize__(self, obj):
        if obj is None: return
        
        super(MetaTime, self).__array_finalize__(obj)
    
    
    def __getitem__(self, idx):
        
        # Try to access as if indices were given
        try:
            return super(MetaTime, self).__getitem__(idx)
        
        # IndexError raised when idx is a datetime64 object
        # TypeError raised when slice contains datetime64 object
        except IndexError as e:
            if type(idx) is int:
                raise
        
        except (TypeError) as e:
            pass
        
        s = self.get_item_index(idx)
        
        return self[s]
    
    
    def get_item_index(self, idx):
        # Turn slice of times into slice of indices
        if type(idx) is slice:
            if (idx.step is not None) & (type(idx.step) is not int):
                raise ValueError('slice.step must be None or int.')
            
            # TODO:
            #   Why is this not handled properly in __getitem__?
            #   Test: t[slice(None, None, None)]
            if idx.start is None:
                i1 = 1
            else:
                i1 = self.get_closest_value(idx.start, greater=True)[1]
            
            if idx.stop is None:
                i2 = len(self)
            else:
                i2 = self.get_closest_value(idx.stop, lesser=True)[1]
            
            s = slice(i1, i2+1, idx.step)
        
        # Turn list of times into list of indices
        elif type(idx) is list:
            units = self.get_units()
            s = [self.get_closest_value(np.datetime64(t, units))[1] for t in idx]
        
        # Do not except tuples (which would be multi-dimensional indexing)
        elif type(idx) is tuple:
            raise ValueError('Indices must be scalar, list, or slice, not tuple.')
        
        # Turn time into index
        else:
            t = np.datetime64(idx, self.get_units())
        
            # Return exact match
            s = self.get_closest_value(t)[1]
#            try:
#                s = np.where(self == t)[0][0]
#            except IndexError:
#                s = self.nearest(t)
        
        return s
    
    
    def nearest(self, time):
        """
        Find the index of the element closest to `time`.

        Parameters
        ----------
        time : `numpy.datetime64`
            Time for which the index of the nearest element is desired.

        Returns
        -------
        idx : int
            Index of the nearest element to `time`.
        """
        # Could be faster: https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item-using-max-min-on-a-list
#        absdiff = np.abs(self-time)
#        return = min(range(len(absdiff)), key=absdiff.__getitem__)
        return np.argmin(np.abs(self-time))
    
    def get_units(self):
        """
        Extract the precision from the dtype attribute.

        Returns
        -------
        units : str
            Precision of the time array.
        """
        try:
            return re.search('\[(.+?)\]', self.dtype.name).group(1)
        except AttributeError:
            return ''
    
    
    def get_closest_value(self, target, greater=False, lesser=False):
        """
        Find closest in time to the target time.

        Parameters
        ----------
        target : `datetime.datetime` or `numpy.datetime64`
            Target time.
        greater : bool, optional
            If ``True``, return the closest value greater than `target`.
            The default is ``False``.
        lesser : bool, optional
            If ``True``, return the closes value less than `target`.
            The default is ``False``.
        

        Returns
        -------
        value : :class:`numpy.datetime64`
            Value closest in time to `target`.
        index : int
            Index of the value closest in time to `target`.
        """
        if greater & lesser:
            raise ValueError('Keywords greater and lesser cannot both be true.')
        
        n = len(self)
        ileft = 0
        iright = n - 1
        imid = 0

        # edge case - last or above all
        if target >= self[n - 1]:
            return self[n - 1], n-1
    
        # edge case - first or below all
        if target <= self[0]:
            return self[0], 0
    
        # BSearch solution: Time & Space: Log(N)
        while ileft < iright:
            imid = (ileft + iright) // 2  # find the mid
            if target < self[imid]:
                iright = imid
            elif target > self[imid]:
                ileft = imid + 1
            else:
                return self[imid]
        
        if target < self[imid]:
            if greater:
                return self[imid], imid
            elif lesser:
                return self[imid-1], imid-1
            else:
                idx = self.find_closest_index(imid-1, imid, target)
                return self[idx], idx
        else:
            if greater:
                return self[imid+1], imid+1
            elif lesser:
                return self[imid], imid
            else:
                idx = self.find_closest_index(imid, imid+1, target)
                return self[idx], idx


    # findClosest
    # We find the closest by taking the difference
    # between the target and both values. It assumes
    # that val2 is greater than val1 and target lies
    # between these two.
    @staticmethod
    def find_closest(val1, val2, target):
        return val2 if target - val1 >= val2 - target else val1
    
    def find_closest_index(self, idx1, idx2, target):
        try:
            return idx2 if target - self[idx1] >= self[idx2] - target else idx1
        except:
            pdb.set_trace()
    

def test_get_item():
    # Initial time array
    t0 = np.datetime64('2013-01-01T12:30:00')
    nsec = 10
    t = [t0 + np.timedelta64(deltat, 's') for deltat in range(nsec)]
    t = MetaTime(t)

    # Scalar index
    #   - Currently returning a datetime64 value. Should return a
    #     MetaTime instance as a scalar array.
    #   - How do I get the superclass's __getitem__ to return a MetaTime?
    print('Scalar Index:')
    result = t[0]
    print(result)
    print(type(result))
    
    # Find nearest time
    print('Time Interval of Interest:')
    t1 = t0 + np.timedelta64(2400, 'ms')
    t2 = t0 + np.timedelta64(8800, 'ms')
    print(t1, t2)
    
    print('Find single time:')
    result = t[t1]
    print(result)
    
    # Find time interval
    print('Finding a slice:')
    result = t[t1:t2]
    print(result)
    
    # Find individual times
    print('Finding list of times:')
    result = t[[t1, t2]]
    print(result)
    
    # Access multiple dimensions
    print('Access multiple dimensions:')
    try:
        result = t[t1, t2]
        print(result)
    except ValueError as e:
        print(e)
    

def test_bisection():
    # Create a test to see if idx = (self == time).any()[0] is faster
    # than the bisection method
    raise NotImplemented


def test_timedelta64():
    # Initial time array
    t0 = np.datetime64('2013-01-01T12:30:00')
    nsec = 10
    t = [t0 + np.timedelta64(deltat, 's') for deltat in range(nsec)]
    t = MetaTime(t)
    
    dt = t - t[0]
    print('type(dt) = {0}, type(dt[0]) = {1}'.format(type(dt), type(dt[0])))

def main1():
    # Create array of times
#    PST = pytz.timezone('America/Los_Angeles')
    t = np.zeros(4, dtype='datetime64[s]')
    t[:] = [dt.datetime(2013, 1, 1, 12, 30, 0),
            dt.datetime(2013, 1, 1, 13, 30, 0),
            dt.datetime(2013, 1, 1, 14, 30, 0),
            dt.datetime(2013, 1, 1, 15, 30, 0)]
    t[:] = ['2013-01-01T12:30:00',
            '2013-01-01T13:30:00',
            '2013-01-01T14:30:00',
            '2013-01-01T15:30:00']
    t = MetaTime(t)
    
    assert isinstance(t[1] - t[0], np.timedelta64), 'Subtraction does not return timedelta64.'
    
    # Name the variable and cache it
    t.name = 'Test Time'
    t.cache()
    assert t.iscached(), 'MetaTime instance is not cached.'
    
    # Check what is in the cache
    pyarray.MetaBase.names()

#    if isinstance(
    
#    ts2 = ts + 1
#    print( ts2, ts2.x, ts2.t, ts2.__class__, ts2.__class__.__bases__)
#    print( ts)
#    print( ts[0], ts[1], ts[2], ts[3])
#    print( ts[datetime(2013,1,1,15,30,0,tzinfo=PST)])
#    print( ts[datetime(2013,1,1,14,45,0,tzinfo=PST)])
#    print( ts.__class__)
#    print( ts.__class__.__bases__)


def main_plot_time():
    from matplotlib import pyplot as plt
    
    # Simple dataset with datetimes
    data = np.array([1,3,2,4])
    time = [dt.datetime(2013, 1, 1, 12, 30, 0),
            dt.datetime(2013, 1, 1, 13, 30, 0),
            dt.datetime(2013, 1, 1, 14, 30, 0),
            dt.datetime(2013, 1, 1, 15, 30, 0)]
    
    # matplotlib correctly interprets/displays the time
    plt.plot(time, data)
    plt.show()
    
    # Now cast as np.datetime64 objects. Time is not displayed as hh:mm:ss
    time = np.array(time, dtype='datetime64[s]')
    plt.plot(time, data)
    plt.show()
    
    # Now cast as MetaTime object. Time is displayed same as datetime64
    time = MetaTime(time)
    plt.plot(time, data)
    plt.show()
    
    # Now recast MetaTime as datetime and plot
    time = time.astype(dt.datetime)
    plt.plot(time, data)
    plt.show()


if __name__ == '__main__':
    main_test()

