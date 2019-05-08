# Instructions for subclassing numpy.ndarray are taken from (9/8/2017)
#   https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
import numpy as np
import datetime as dt
import pyarray
import pytz
import pdb

class MetaTime(pyarray.MetaBase):
    
    def __new__(cls, x, units='us', **kwargs):
        # Create a datetime64 object
        obj = super(MetaTime, cls).__new__(cls, x, dtype='datetime64['+units+']', **kwargs)
        
        # Finally, we must return the newly created object:
        return obj

    
    def __array_finalize__(self, obj):
        if obj is None: return
        
        super(MetaTime, self).__array_finalize__(obj)
    
    
    def __getitem__(self, idx):
        obj = super(MetaTime, self).__getitem__(idx)
        return obj
    
    
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
    
    pdb.set_trace()
    
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

