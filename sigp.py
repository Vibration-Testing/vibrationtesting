"""
Created on 14 May 2015
@author: Joseph C. Slater
"""
__license__ = "Joseph C. Slater"

__docformat__ = 'reStructuredText'


import math
import warnings

import numpy as np
from numpy import ma
import scipy as sp
import scipy.signal as sig
import scipy.fftpack as fftpack
from numpy import linalg as la 
import matplotlib
import matplotlib.pyplot as plt
rcParams = matplotlib.rcParams

import matplotlib.cbook as cbook
from matplotlib.cbook import _string_to_bool, mplDeprecation
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.contour as mcontour
import matplotlib.dates as _  # <-registers a date unit converter
from matplotlib import docstring
import matplotlib.image as mimage
import matplotlib.legend as mlegend
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import matplotlib.mlab as mlab
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.quiver as mquiver
import matplotlib.stackplot as mstack
import matplotlib.streamplot as mstream
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.tri as mtri
import matplotlib.transforms as mtrans
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer
from matplotlib.axes._base import _AxesBase
from matplotlib.axes._base import _process_plot_format
import pylab as pl

# Notes:
# ------
#
# In order to be consistent with the Control Systems Library, increasing time
# or increasing frequency steps positively with increased column number (second dimension). Rows (first dimension)
# correspond to appropriate degress of freedom, output numbers, etc. The
# third dimension indexes each data instance (experiment).
#
# Awesome: I don't know which standard the package is following now!
# In order to be consistent with scipy and matlab, increasing time or frequency indices
# increases in the 0th dimension (0,1,2). The first dimension is the index.
# The second dimension is the realization number (for multiple runs/simulations/datasets).
#


def window(x, windowname = 'hanning', normalize = False):
    """returns w
    Create a  window of length :math:`x`, or a hanning window sized to match :math:`x`
    such that x*w is the windowed result.
    
    Parameters
 
    x:                 1) Integer. Number of points in desired hanning windows.
                       2) Array to which window needs to be applied. 
    windowname:        One of: hanning, hamming, blackman, flatwin, boxwin
    normalize (False): Adjust power level (for use in ASD) to 1

    Returns
 
    w: 1) hanning window array of size x
       2) windowing array. Windowed array is then x*w

    :Example:

    >>> import numpy as np
    >>> import vibrationtesting as vt
    >>> import matplotlib.pyplot as plt
    >>> sample_freq = 1e3
    >>> tfinal = 5
    >>> fs = 100
    >>> A = 10
    >>> freq = 5
    >>> noise_power = 0.001 * sample_freq / 2
    >>> time = np.reshape(np.arange(0, tfinal, 1/sample_freq),(1,-1))
    >>> xsin = A*np.sin(2*np.pi*freq*time)
    >>> xcos = A*np.cos(2*np.pi*freq*time)
    # assembling individual records. vstack assembles channels
    >>> x=np.dstack((xsin,xcos)) # assembling individual records. vstack
    >>> xw=vt.hanning(x)*x
    >>> plt.subplot(2, 1, 1)
    <matplotlib.axes._subplots.AxesSubplot object at ...>
    >>> plt.plot(time.T,x.T)
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.ylim([-20, 20])
    (-20, 20)
    >>> plt.title('Unwindowed data, 2 records.')
    <matplotlib.text.Text object at ...>
    >>> plt.ylabel('$x(t)$')
    <matplotlib.text.Text object at ...>
    >>> plt.subplot(2, 1, 2)
    <matplotlib.axes._subplots.AxesSubplot object at ...>
    >>> plt.title('Original (raw) data.')
    <matplotlib.text.Text object at ...>
    >>> plt.plot(time[0,:],xw[0,:],time[0,:],vt.hanning(x)[0,:]*A,'--',time[0,:],-vt.hanning(x)[0,:]*A,'--')
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.ylabel('Hanning windowed $x(t)$')
    <matplotlib.text.Text object at ...>
    >>> plt.xlabel('time')
    <matplotlib.text.Text object at ...>
    >>> plt.title('Effect of window. Note the scaling to conserve ASD amplitude')
    <matplotlib.text.Text object at ...>
    >>> plt.show()
    
    """
    
    if isinstance(x,(list, tuple, np.ndarray)):
        # Create Hanning windowing array of dimension n by N by nr
        # where N is number of data points and n is the number of number of inputs or outputs.
        # and nr is the number of records.
        
        swap = 0
        if len(x.shape) == 1:
            # We have either a scalar or 1D array
            if x.shape[0] == 1:
                print(" x is a scalar... and shouldn\'t have entered this part of the loop.")
            else:
                N = len(x)
            
            f = self.window(N , windowname = windowname)
            
        elif len(x.shape)==3:
            
            if x.shape[0]>x.shape[1]:
                x = sp.swapaxes(x,0,1)
                swap = 1
                print('Swapping axes temporarily to be compliant with expectations. I\'ll fix them in your result')

            f=window(N , windowname = windowname)
            f,_, _=np.meshgrid(f, np.arange(x.shape[0]),np.arange(x.shape[2]))
            if swap == 1:
                f = sp.swapaxes(f,0,1)
            
        elif len(x.shape)==2:
            #f,_=np.meshgrid(f[0,:],np.arange(x.shape[0]))
            #print('b')

            if x.shape[0]>x.shape[1]:
                x = sp.swapaxes(x,0,1)
                swap = 1
                print('Swapping axes temporarily to be compliant with expectations. I\'ll fix them in your result')

            f = window(x.shape[1] , windowname = windowname)
            f, _ = np.meshgrid(f, np.arange(x.shape[0]))
            if swap == 1:
                f = sp.swapaxes(f,0,1)
            
    else:
        #print(x)
        # Create hanning window of length x
        N = x
        #print(N)
        if windowname is 'hanning':
            f = np.sin(np.pi * np.arange(N)/(N-1))**2 * np.sqrt(8/3)
        elif windowname is 'hamming':
            f=(0.54-0.46*np.cos(2*np.pi*(np.arange(N))/(N-1)))*np.sqrt(5000/1987)
        elif windowname is 'blackman':
            print('blackman')
            f=(0.42-0.5*np.cos(2*np.pi*(np.arange(N)+.5)/(N))+.08*np.cos(4*np.pi*(np.arange(N)+.5)/(N)))*np.sqrt(5000/1523)
        elif windowname is 'flatwin':
            f=1.0-1.933*np.cos(2*np.pi*(np.arange(N))/(N-1))+1.286*np.cos(4*np.pi*(np.arange(N))/(N-1))-0.338*np.cos(6*np.pi*(np.arange(N))/(N-1))+0.032*np.cos(8*np.pi*(np.arange(N))/(N-1))
        elif windowname is 'boxwin':
            f=np.ones((1,N))
        else:
            f = np.ones((1,N))
            print("I don't recognize that window name. Sorry")
            
        if normalize is True:
            f = f/np.linalg.norm(f) * np.sqrt(N)
    return f


def hanning(x, normalize = False):
    """returns w
    Create a hanning window of length :math:`x`, or a hanning window sized to match :math:`x`
    such that x*w is the windowed result.
    
    Parameters
 
    x:                 1) Integer. Number of points in desired hanning windows.
                       2) Array to which window needs to be applied. 
    normalize (False): Adjust power level (for use in ASD) to 1

    Returns
 
    w: 1) hanning window array of size x
       2) windowing array. Windowed array is then x*w

    :Example:

    >>> import numpy as np
    >>> import vibrationtesting as vt
    >>> import matplotlib.pyplot as plt
    >>> sample_freq = 1e3
    >>> tfinal = 5
    >>> fs = 100
    >>> A = 10
    >>> freq = 5
    >>> noise_power = 0.001 * sample_freq / 2
    >>> time = np.reshape(np.arange(0, tfinal, 1/sample_freq),(1,-1))
    >>> xsin = A*np.sin(2*np.pi*freq*time)
    >>> xcos = A*np.cos(2*np.pi*freq*time)
    # assembling individual records. vstack assembles channels
    >>> x=np.dstack((xsin,xcos)) # assembling individual records. vstack
    >>> xw=vt.hanning(x)*x
    >>> plt.subplot(2, 1, 1)
    <matplotlib.axes._subplots.AxesSubplot object at ...>
    >>> plt.plot(time.T,x.T)
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.ylim([-20, 20])
    (-20, 20)
    >>> plt.title('Unwindowed data, 2 records.')
    <matplotlib.text.Text object at ...>
    >>> plt.ylabel('$x(t)$')
    <matplotlib.text.Text object at ...>
    >>> plt.subplot(2, 1, 2)
    <matplotlib.axes._subplots.AxesSubplot object at ...>
    >>> plt.title('Original (raw) data.')
    <matplotlib.text.Text object at ...>
    >>> plt.plot(time[0,:],xw[0,:],time[0,:],vt.hanning(x)[0,:]*A,'--',time[0,:],-vt.hanning(x)[0,:]*A,'--')
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.ylabel('Hanning windowed $x(t)$')
    <matplotlib.text.Text object at ...>
    >>> plt.xlabel('time')
    <matplotlib.text.Text object at ...>
    >>> plt.title('Effect of window. Note the scaling to conserve ASD amplitude')
    <matplotlib.text.Text object at ...>
    >>> plt.show()
    
    """
    
    if isinstance(x,(list, tuple, np.ndarray)):
        # Create Hanning windowing array of dimension n by N by nr
        # where N is number of data points and n is the number of number of inputs or outputs.
        # and nr is the number of records.
        print(len(x.shape))
        swap = 0
        if len(x.shape) == 1:
            # We have either a scalar or 1D array
            if x.shape[0] == 1:
                print(" x is a scalar... and shouldn\'t have entered this part of the loop.")
            else:
                N = len(x)
            f = hanning(N)
            
        elif len(x.shape)==3:
            #print('a')
            #print(f.shape)
            
            if x.shape[0]>x.shape[1]:
                x = sp.swapaxes(x,0,1)
                swap = 1
                print('Swapping axes temporarily to be compliant with expectations. I\'ll fix them in your result')

            f=hanning(x.shape[1])
            f,_, _=np.meshgrid(f, np.arange(x.shape[0]),np.arange(x.shape[2]))
            if swap == 1:
                f = sp.swapaxes(f,0,1)
            
        elif len(x.shape)==2:
            #f,_=np.meshgrid(f[0,:],np.arange(x.shape[0]))
            #print('b')
            print('length = 2')
            print(x.shape)
            if x.shape[0]>x.shape[1]:
                x = sp.swapaxes(x,0,1)
                swap = 1
                print('Swapping axes temporarily to be compliant with expectations. I\'ll fix them in your result')

            f=hanning(x.shape[1])
            f, _ = np.meshgrid(f, np.arange(x.shape[0]))
            if swap == 1:
                f = sp.swapaxes(f,0,1)
            
    else:
        #print(x)
        # Create hanning window of length x
        N = x
        #print(N)
        f = np.sin(np.pi * np.arange(N)/(N-1))**2 * np.sqrt(8/3)
        if normalize is True:
            f = f/np.linalg.norm(f) * np.sqrt(N)
    return f


def blackwin(x):
    """
    w=blackwin(n)
    Return the n point Blackman window
    x_windows=blackwin(x)
    Returns x as the Blackman windowing array x_window
    The windowed signal is then x*x_window
    """
    print('blackwin is untested')
    if isinstance(x,(list, tuple, np.ndarray)):
        n=x.shape[1]
        f=blackwin(n)
                
        if len(x.shape)==3:
            f,_,_=np.meshgrid(f[0,:],np.arange(x.shape[0]),np.arange(x.shape[2]))
        else:
            f,_=np.meshgrid(f[0,:],np.arange(x.shape[0]))
    else:

        n=x
        f=np.reshape((0.42-0.5*np.cos(2*np.pi*(np.arange(n)+.5))/(n)+.08*np.cos(4*np.pi*(np.arange(n)+.5))/(n))*np.sqrt(5000/1523),(1,-1))
        f=f/np.linalg.norm(f)*np.sqrt(n)


    return f
def expwin(x,ts=.75):
    """
    w=expwin(n)
    Return the n point exponential window
    x_windows=expwin(x)
    Returns x as the expwin windowing array x_windowed
    The windowed signal is then x*x_window
    The optional second argument set the 5% "settling time" of the window. Default is ts=0.75 
    """
    print('expwin is untested')
    tc=-ts/np.log(.05)
    if isinstance(x,(list, tuple, np.ndarray)):
        n=x.shape[1]
        f=expwin(n)
                
        if len(x.shape)==3:
            f,_,_=np.meshgrid(f[0,:],np.arange(x.shape[0]),np.arange(x.shape[2]))
        else:
            f,_=np.meshgrid(f[0,:],np.arange(x.shape[0]))
    else:
        n=x
        v=(n-1)/n*np.arange(n)+(n-1)/n/2
        f=exp(-v/tc/(n-1))
        f=f/np.linalg.norm(f)*np.sqrt(n)    
        f=np.reshape(f,(1,-1))
        f=f/np.linalg.norm(f)*np.sqrt(n)


    return f
def hammwin(x):
    """
    w=hammwin(n)
    Return the n point hamming window
    x_windows=hamming(x)
    Returns x as the hamming windowingarray x_windowed
    The windowed signal is then x*x_window
    """
    print('hammwin is untested')
    if isinstance(x,(list, tuple, np.ndarray)):
        n=x.shape[1]
        f=hammwin(n)
                
        if len(x.shape)==3:
            f,_,_=np.meshgrid(f[0,:],np.arange(x.shape[0]),np.arange(x.shape[2]))
        else:
            f,_=np.meshgrid(f[0,:],np.arange(x.shape[0]))
    else:

        n=x
        f=np.reshape((0.54-0.46*np.cos(2*np.pi*(np.arange(n))/(n-1)))*np.sqrt(5000/1987),(1,-1))
        f=f/np.linalg.norm(f)*np.sqrt(n)


    return f
def flatwin(x):
    """
    w=flatwin(n)
    Return the n point flat top window
    x_windows=flatwin(x)
    Returns x as the flat top windowing array x_windowed
    The windowed signal is then x*x_window
    McConnell, K. G., "Vibration Testing: Theory and Practice," Wiley, 1995.
    """
    print('flatwin is untested')
    if isinstance(x,(list, tuple, np.ndarray)):
        n=x.shape[1]
        f=flatwin(n)
                
        if len(x.shape)==3:
            f,_,_=np.meshgrid(f[0,:],np.arange(x.shape[0]),np.arange(x.shape[2]))
        else:
            f,_=np.meshgrid(f[0,:],np.arange(x.shape[0]))
    else:

        n=x
        f=np.reshape((1.0-1.933*np.cos(2*np.pi*(np.arange(n))/(n-1))+1.286*np.cos(4*np.pi*(np.arange(n))/(n-1))-0.338*np.cos(6*np.pi*(np.arange(n))/(n-1))+0.032*np.cos(8*np.pi*(np.arange(n))/(n-1))),(1,-1))
        f=f/np.linalg.norm(f)*np.sqrt(n)
        
    return f
def boxwin(x):
    """
    w=boxwin(n)
    Return the n point box window (uniform)
    x_windows=boxwin(x)
    Returns x as the boxwin windowing array x_windowed
    The windowed signal is then x*x_window
    """
    print('boxwin is untested')
    if isinstance(x,(list, tuple, np.ndarray)):
        n=x.shape[1]
        f=boxwin(n)
                
        if len(x.shape)==3:
            f,_,_=np.meshgrid(f[0,:],np.arange(x.shape[0]),np.arange(x.shape[2]))
        else:
            f,_=np.meshgrid(f[0,:],np.arange(x.shape[0]))
    else:

        n=x
        #f=np.reshape((1.0-1.933*np.cos(2*np.pi*(np.arange(n))/(n-1))+1.286*np.cos(4*np.pi*(np.arange(n))/(n-1))-0.338*np.cos(6*np.pi*(np.arange(n))/(n-1))+0.032*np.cos(8*np.pi*(np.arange(n))/(n-1))),(1,-1))
        f=np.reshape(np.ones((1,n)),(1,-1))
        f=f/np.linalg.norm(f)*np.sqrt(n)
        
    return f


def hannwin(x):
    f=hanning(x)
    return f
def asd(x,t,window="hanning",ave=bool(True)):
    """
    Calculate the autospectrum (power spectrum) density of a signal x

    :Example:

    >>> from scipy import signal
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import vibrationtesting as vt
    >>> from numpy import linalg

    Generate a 5 second test signal, a 10 V sine wave at 50 Hz, corrupted by
    0.001 V**2/Hz of white noise sampled at 1 kHz.

    >>> sample_freq = 1e3
    >>> tfinal = 5
    >>> sig_freq=50
    >>> A=10
    >>> noise_power = 0.0001 * sample_freq / 2
    >>> noise_power = A/1e12
    >>> time = np.arange(0,tfinal,1/sample_freq)
    >>> time = np.reshape(time, (1, -1))
    >>> x = A*np.sin(2*np.pi*sig_freq*time)
    >>> x = x + np.random.normal(scale=np.sqrt(noise_power), size=(1, time.shape[1]))
    >>> plt.subplot(2,1,1)
    <matplotlib...>
    >>> plt.plot(time[0,:],x[0,:])
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.title('Time history')
    <matplotlib...>
    >>> plt.xlabel('Time (sec)')
    <matplotlib...>
    >>> plt.ylabel('$x(t)$')
    <matplotlib...>

    Compute and plot the autospectrum density.

    >>> freq_vec, Pxx = vt.asd(x, time, window="hanning", ave=bool(False))
    >>> plt.subplot(2,1,2)
    <matplotlib...>
    >>> plt.plot(freq_vec[0,:], 20*np.log10(Pxx[0,:,:]))
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.ylim([-400, 100])
    (-400, 100)
    >>> plt.xlabel('frequency [Hz]')
    <matplotlib.text.Text object at ...>
    >>> plt.ylabel('PSD [V**2/Hz]')
    <matplotlib.text.Text object at ...>
    >>> plt.show()

    If we average the last half of the spectral density, to exclude the
    peak, we can recover the noise power on the signal.


    Now compute and plot the power spectrum.

    """
    f, Pxx=crsd(x,x,t,window,ave)
    Pxx=Pxx.real
    return f, Pxx

def crsd(x, y, t, windowname = "hanning", ave = bool(True)):
    """
    Calculate the cross spectrum (power spectrum) density between two signals.

    :Example:

    >>> from scipy import signal
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import vibrationtesting as vt
    >>> from numpy import linalg

    Generate a 5 second test signal, a 10 V sine wave at 50 Hz, corrupted by
    0.001 V**2/Hz of white noise sampled at 1 kHz.

    >>> sample_freq = 1e3
    >>> tfinal = 5
    >>> sig_freq=50
    >>> A=10
    >>> noise_power = 0.0001 * sample_freq / 2
    >>> noise_power = A/1e12
    >>> time = np.arange(0,tfinal,1/sample_freq)
    >>> time = np.reshape(time, (1, -1))
    >>> x = A*np.sin(2*np.pi*sig_freq*time)
    >>> x = x + np.random.normal(scale=np.sqrt(noise_power), size=(1, time.shape[1]))
    >>> plt.subplot(2,1,1)
    <matplotlib...>
    >>> plt.plot(time[0,:],x[0,:])
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.title('Time history')
    <matplotlib...>
    >>> plt.xlabel('Time (sec)')
    <matplotlib...>
    >>> plt.ylabel('$x(t)$')
    <matplotlib...>

    Compute and plot the autospectrum density.

    >>> freq_vec, Pxx = vt.asd(x, time, window="hanning", ave=bool(False))
    >>> plt.subplot(2,1,2)
    <matplotlib...>
    >>> plt.plot(freq_vec[0,:], 20*np.log10(Pxx[0,:,:]))
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.ylim([-400, 100])
    (-400, 100)
    >>> plt.xlabel('frequency [Hz]')
    <matplotlib.text.Text object at ...>
    >>> plt.ylabel('PSD [V**2/Hz]')
    <matplotlib.text.Text object at ...>
    >>> plt.show()

    If we average the last half of the spectral density, to exclude the
    peak, we can recover the noise power on the signal.


    Now compute and plot the power spectrum.
    """
    #t=np.reshape(t,(1,-1))
    
    if t.shape[0]>1:
        dt=t[2]-t[1]
    elif t.shape[1]>1:
        dt=t[2]-t[1]
        print('t must be a scalar or size (n,1)')
    elif t.shape[1]==1 and t.shape[0]==1:
        dt=t[0]
    if dt <= 0:
        print('You sent in bad data. Delta t is negative. Please check your inputs.')

    if len(x.shape)==1:
        x = sp.expand_dims(x, axis = 0)
        x = sp.expand_dims(x, axis = 2)
        y = sp.expand_dims(y, axis = 0)
        y = sp.expand_dims(y, axis = 2)
    n=x.shape[1];

    # No clue what this does, and I wrote it. Comment your code, you fool!
    # What this "should" do is assure that the data is longer in 0 axis than the others. 
    # if len(x.shape)==2:
    #     # The issue fixed here is that the user put time along the 1 axis (instead of zero)
    #     if (x.shape).index(max(x.shape))==0:
    #         #x=x.reshape(max(x.shape),-1,1)
    #         print('I think you put time along the 0 axis instead of the 1 axis. Not even attempting to fix this.')
    #     else:
    #         # Here we are appending a 3rd dimension to simplify averaging command later. We could bypass at that point, and should. 
    #         x=x.reshape(max(x.shape),-1,1)

    # if len(y.shape)==2:
    #     if (y.shape).indey(may(y.shape))==0:
    #         #y=y.reshape(may(y.shape),-1,1)
    #         print('I think you put time along the 0 axis instead of the 1 axis. Not attempting to fix this.')            
    #     else:
    #         y=y.reshape(may(y.shape),-1,1)
    # # Should use scipy.signal windows. I need to figure this out. Problem is: They don't scale the ASDs by the windowing "weakening". 
    
    if window=="none":
        a=1
    else:
        print('This doesn\'t work yet')
        win=1
        if window=="hanning":#BLACKWIN, BOXWIN, EXPWIN, HAMMWIN, FLATWIN and TRIWIN
            #print('shape of x')
            #print(x.shape)
            win=window(x, windowname = 'hanning')
        elif window=="blackwin":
            win=window(x, windowname = 'blackwin')
        elif window=="boxwin":
            win=window(x, windowname = 'boxwin')
        elif window=="expwin":
            win=window(x, windowname = 'expwin')
        elif window=="hammwin":
            win=window(x, windowname = 'hamming')
        elif window=="triwin":
            win=window(x, windowname = 'triwin')
        elif window=="flatwin":
            win=window(x, windowname = 'flatwin')
        
        y=y*win
        x=x*win
        del win

    
    ffty=np.fft.rfft(y,axis = 1)*dt

    fftx=np.fft.rfft(x,n,axis = 1)*dt

    Pxy=np.conj(fftx)*ffty/(n*dt)*2

    
    if len(Pxy.shape)==3 and Pxy.shape[2]>1 and ave:
        Pxy=np.mean(Pxy,2)
        
    
    nfreq=1/dt/2;
    f=np.linspace(0, nfreq, Pxy.shape[1])
    
   
    return f, Pxy

def frfest(x, f, dt, window="hanning", ave=bool(True), Hv=bool(False)):#,n,options)
    """returns freq, H1, H2, coh, Hv
    
    Estimates the :math:`H(j\\omega)` Frequency Response Functions (FRFs) between :math:`x` and :math:`f`.

        - parameters using ``:param <name>: <description>``
        - type of parameters ``:type <name>: <description>``
        - returns using ``:returns: <description>``
        - examples (doctest)
        - seealso using ``.. seealso:: text``
        - notes using ``.. note:: text``
        - warning using ``.. warning:: text``
        - todo ``.. todo:: text``
    

    :param x: output or response of system
    :param f: input to system
    :param dt: time step or time array
    :param window: name of data window to apply
    :param ave: apply averaging
    :param Hv: calculate :math:`H_v` Frequency Response Function Estimation
    :type x: float array
    :type f: float array
    :type dt: float
    :type window: string
    :type ave: Boolean
    :type Hv: Boolean
    :returns: freq, H1, H2, coh, Hv
    :return freq: frequency vector
    :type freq: float array
    :rtype: float array, float array, float array, float array, float array

    Currently ``window`` and ``ave`` are locked to default values.

    :Example:
        
    >>> import control as ctrl
    >>> import matplotlib.pyplot as plt
    >>> import vibrationtesting as vt
    >>> import numpy as np
    >>> sample_freq = 1e3
    >>> noise_power = 0.001 * sample_freq / 2
    >>> A = np.array([[0, 0, 1, 0],\
              [0, 0, 0, 1],\
              [-200, 100, -.2, .1],\
              [100, -200, .1, -.2]])
    >>> B = np.array([[0], [0], [1], [0]])
    >>> C = np.array([[35, 0, 0, 0], [0, 35, 0, 0]])
    >>> D = np.array([[0], [0]])
    >>> sys = ctrl.ss(A, B, C, D)
    >>> tin = np.arange(0, 51.2, .1)
    >>> nr=.5   # 0 is all noise on input
    >>> for i in np.arange(5): #was 2*50
    ...     u = np.random.normal(scale=np.sqrt(noise_power), size=tin.shape)
    ...     #print(u)
    ...     t, yout, xout = ctrl.forced_response(sys, tin, u,rtol=1e-12,transpose=True)
    ...     if 'Yout' in locals():
    ...         Yout=np.dstack((Yout,yout+nr*np.random.normal(scale=.050*np.std(yout[0,:]), size=yout.shape)))
    ...         Ucomb=np.dstack((Ucomb,u+(1-nr)*np.random.normal(scale=.05*np.std(u), size=u.shape)))
    ...     else:
    ...         Yout=yout+nr*np.random.normal(scale=.050*np.std(yout[0,:]), size=yout.shape) # 5% half the noise on output as on input
    ...         Ucomb=u+(1-nr)*np.random.normal(scale=.05*np.std(u), size=u.shape)#(1, len(tin))) #10% noise signal on input
    >>> plt.plot(tin,Yout[:,0,:])
    >>> Yout=Yout*np.std(Ucomb)/np.std(Yout)#40
    [<matplotlib.lines.Line2D object at ...]
    >>> plt.title('time response')
    <matplotlib.text.Text object...>
    >>> plt.show()
    >>> freq_vec, Pxx = vt.asd(Yout, tin, window="hanning", ave=bool(False))
    >>> plt.plot(freq_vec[0,:], 20*np.log10(Pxx[0,:,:]))
    [<matplotlib.lines.Line2D object at ...]
    >>> plt.title('Raw ASDs')
    <matplotlib.text.Text...>
    >>> plt.show()
    >>> freq_vec, Pxx = vt.asd(Yout, tin, window="hanning", ave=bool(True))
    >>> plt.plot(freq_vec[0,:], 20*np.log10(Pxx[0,:]))
    [<matplotlib.lines.Line2D object at ...]
    >>> plt.title('Averaged ASDs')
    <matplotlib...>
    >>> plt.show()
    >>> f, Txy1, Txy2, coh, Txyv = vt.frfest(Yout, Ucomb, t,Hv=bool(True))
    >>> #fig_amp,=plt.plot(f[0,:],20*np.log10(np.abs(Txy1[0,:])),legend='$H_1$',f[0,:],20*np.log10(np.abs(Txy2[0,:])),legend='$H_2$',f[0,:],20*np.log10(np.abs(Txyv[0,:])),legend='$H_v$')
    >>> (line1, line2, line3) = plt.plot(f[0,:],20*np.log10(np.abs(Txy1[0,:])),f[0,:],20*np.log10(np.abs(Txy2[0,:])),f[0,:],20*np.log10(np.abs(Txyv[0,:]))) 
    >>> plt.title('FRF of ' + str(Yout.shape[2]) + ' averages.')
    <matplotlib.text.Text object at ...>
    >>> plt.legend((line1,line2,line3),('$H_1$','$H_2$','$H_v$'))
    <matplotlib.legend.Legend object ...>
    >>> plt.show()
    >>> plt.plot(f[0,:],180.0/np.pi*np.unwrap(np.angle(Txy1[0,:])),f[0,:],180.0/np.pi*np.unwrap(np.angle(Txy2[0,:])),f[0,:],180.0/np.pi*np.unwrap(np.angle(Txyv[0,:]))) 
    [<matplotlib.lines.Line2D object at ...]
    >>> plt.title('FRF of ' + str(Yout.shape[2]) + ' averages.')
    <matplotlib.text.Text object at ...>
    >>> plt.show()
    >>> plt.plot(f[0,:],coh[0,:])
    [<matplotlib.lines.Line2D object at...
    >>> plt.show()
    >>> vt.frfplot(f,Txy1,freq_max=3.5)
    

    Copyright 1994 by Joseph C. Slater

    :Modifications:
   
    7/6/00: Changed default FRF calculation from H2 to H1, Added H1, H2, and Hv options.
    4/13/15: Converted to Python
    
    .. note:: Not comptible with scipy.signal functions
    .. seealso:: :func:`asd`, :func:`crsd`, :func:`frfplot`.
    .. warning:: hanning window cannot be selected yet. Averaging cannot be unslected yet.
    .. todo:: Fix averaging, windowing, multiple input.
    """


    if len(f.shape)==2:
        if (f.shape).index(max(f.shape))==0:
            f=f.reshape(max(f.shape),min(f.shape),1)
        else:
            f=f.reshape(1,max(f.shape),min(f.shape))

    if len(x.shape)==2:
        if (x.shape).index(max(x.shape))==0:
            x=x.reshape(max(x.shape),min(x.shape),1)
        else:
            x=x.reshape(1,max(x.shape),min(x.shape))

            
    # Note: Two different ways to ignore returned values shown
    Pff = asd(f,dt)[1]
    print('works until here?')
    print(x.shape)
    print(f.shape)
    freq, Pxf = crsd(x, f, dt)
    _,Pxx=asd(x,dt)

    Txf1=np.conj(Pxf/Pff)  # Note Pfx=conj(Pxf) is applied in the H1 FRF estimation
    Txf2=Pxx/Pxf
    Txfv=Txf1*0  # Nulled to avoid output problems/simplify calls if unrequested
    
    coh=(Pxf*np.conj(Pxf)).real/Pxx/Pff

    if Hv:
        import numpy.linalg as la 
        for i in np.arange(Pxx.shape[1]):
            frfm=np.array([[Pff[0,i], np.conj(Pxf[0,i])],[Pxf[0,i], Pxx[0,i]]]);
            #print('index number ' + str(i))
            #print(frfm)
            alpha=1#np.sqrt(Pff[0,i]/Pxx[0,i])
            #print(alpha)
            frfm=np.array([[Pff[0,i], alpha*np.conj(Pxf[0,i])],[alpha*Pxf[0,i], alpha**2*Pxx[0,i]]]);
            #print('new frfm')
            #print(frfm)
            lam,vecs=la.eigh(frfm)
            #print(lam)
            #print(vecs)
            index=lam.argsort()
            #print(index)
            lam=lam[index]
            #print(lam)
            vecs=vecs[:,index]
            #print(vecs)
            #print(np.array([Txf1[0,i], -(vecs[0,0]/vecs[1,0]), -(vecs[1,0]/vecs[1,1]), Txf2[0,i]]))
            Txfv[0,i]=-(vecs[0,0]/vecs[1,0])/alpha#*np.sqrt(alpha)
            a=1
            b=-Pxx[0,i]-Pff[0,i]
            c=Pxx[0,i]*Pff[0,i]-abs(Pxf[0,i])**2
            #lambda1=((Pxx[0,i]+Pff[0,i])-np.sqrt((Pxx[0,i]+Pff[0,i])*2+4*((Pxx[0,i]*Pff[0,i]-abs(Pxf[0,i])**2))))/2
            lambda1=(-b-np.sqrt(b**2-4*a*c))/2/a
            #Txfv[0,i]=np.conj(Pxf[0,i])/(Pff[0,i]-lambda1)*alpha
            #print(Txfv[0,i])
            #person = input('Next point: ')
            #print(np.dot(frfm, vecs[:, 0]) - lam[0] * vecs[:, 0])
            #print(np.dot(frfm, vecs[:, 1]) - lam[1] * vecs[:, 1])
            
    return freq, Txf1, Txf2, coh, Txfv


    ## def acorr(self, x, **kwargs):
    ##     """
    ##     Plot the autocorrelation of `x`.

    ##     Parameters
    ##     ----------

    ##     x : sequence of scalar

    ##     hold : boolean, optional, default: True

    ##     detrend : callable, optional, default: `mlab.detrend_none`
    ##         x is detrended by the `detrend` callable. Default is no
    ##         normalization.

    ##     normed : boolean, optional, default: True
    ##         if True, normalize the data by the autocorrelation at the 0-th
    ##         lag.

    ##     usevlines : boolean, optional, default: True
    ##         if True, Axes.vlines is used to plot the vertical lines from the
    ##         origin to the acorr. Otherwise, Axes.plot is used.

    ##     maxlags : integer, optional, default: 10
    ##         number of lags to show. If None, will return all 2 * len(x) - 1
    ##         lags.

    ##     Returns
    ##     -------
    ##     (lags, c, line, b) : where:

    ##       - `lags` are a length 2`maxlags+1 lag vector.
    ##       - `c` is the 2`maxlags+1 auto correlation vectorI
    ##       - `line` is a `~matplotlib.lines.Line2D` instance returned by
    ##         `plot`.
    ##       - `b` is the x-axis.

    ##     Other parameters
    ##     -----------------
    ##     linestyle : `~matplotlib.lines.Line2D` prop, optional, default: None
    ##         Only used if usevlines is False.

    ##     marker : string, optional, default: 'o'

    ##     Notes
    ##     -----
    ##     The cross correlation is performed with :func:`numpy.correlate` with
    ##     `mode` = 2.

    ##     Examples
    ##     --------

    ##     `~matplotlib.pyplot.xcorr` is top graph, and
    ##     `~matplotlib.pyplot.acorr` is bottom graph.

    ##     .. plot:: mpl_examples/pylab_examples/xcorr_demo.py

    ##     """
    ##     return self.xcorr(x, x, **kwargs)

    ## @docstring.dedent_interpd
    ## def xcorr(self, x, y, normed=True, detrend=mlab.detrend_none,
    ##           usevlines=True, maxlags=10, **kwargs):
    ##     """
    ##     Plot the cross correlation between *x* and *y*.

    ##     Parameters
    ##     ----------

    ##     x : sequence of scalars of length n

    ##     y : sequence of scalars of length n

    ##     hold : boolean, optional, default: True

    ##     detrend : callable, optional, default: `mlab.detrend_none`
    ##         x is detrended by the `detrend` callable. Default is no
    ##         normalization.

    ##     normed : boolean, optional, default: True
    ##         if True, normalize the data by the autocorrelation at the 0-th
    ##         lag.

    ##     usevlines : boolean, optional, default: True
    ##         if True, Axes.vlines is used to plot the vertical lines from the
    ##         origin to the acorr. Otherwise, Axes.plot is used.

    ##     maxlags : integer, optional, default: 10
    ##         number of lags to show. If None, will return all 2 * len(x) - 1
    ##         lags.

    ##     Returns
    ##     -------
    ##     (lags, c, line, b) : where:

    ##       - `lags` are a length 2`maxlags+1 lag vector.
    ##       - `c` is the 2`maxlags+1 auto correlation vectorI
    ##       - `line` is a `~matplotlib.lines.Line2D` instance returned by
    ##         `plot`.
    ##       - `b` is the x-axis (none, if plot is used).

    ##     Other parameters
    ##     -----------------
    ##     linestyle : `~matplotlib.lines.Line2D` prop, optional, default: None
    ##         Only used if usevlines is False.

    ##     marker : string, optional, default: 'o'

    ##     Notes
    ##     -----
    ##     The cross correlation is performed with :func:`numpy.correlate` with
    ##     `mode` = 2.
    ##     """

    ##     Nx = len(x)
    ##     if Nx != len(y):
    ##         raise ValueError('x and y must be equal length')

    ##     x = detrend(np.asarray(x))
    ##     y = detrend(np.asarray(y))

    ##     c = np.correlate(x, y, mode=2)

    ##     if normed:
    ##         c /= np.sqrt(np.dot(x, x) * np.dot(y, y))

    ##     if maxlags is None:
    ##         maxlags = Nx - 1

    ##     if maxlags >= Nx or maxlags < 1:
    ##         raise ValueError('maglags must be None or strictly '
    ##                          'positive < %d' % Nx)

    ##     lags = np.arange(-maxlags, maxlags + 1)
    ##     c = c[Nx - 1 - maxlags:Nx + maxlags]

    ##     if usevlines:
    ##         a = self.vlines(lags, [0], c, **kwargs)
    ##         b = self.axhline(**kwargs)
    ##     else:

    ##         kwargs.setdefault('marker', 'o')
    ##         kwargs.setdefault('linestyle', 'None')
    ##         a, = self.plot(lags, c, **kwargs)
    ##         b = None
    ##     return lags, c, a, b
def  frfplt(freq,H,freq_min=0,freq_max=0,FLAG=1):
    """returns
    
    Plots frequency response functions in a variety of formats
    
        - parameters using ``:param <name>: <description>``
        - type of parameters ``:type <name>: <description>``
        - returns using ``:returns: <description>``
        - examples (doctest)
        - seealso using ``.. seealso:: text``
        - notes using ``.. note:: text``
        - warning using ``.. warning:: text``
        - todo ``.. todo:: text``
    

    :param freq: frequency data (Hz) of shape (1,n_points)
    :param H: Frequency Response Functions, shape (n,n_points)
    :param freq_min: lowest frequency to plot
    :param freq_min: highest frequency to plot
    :param FLAG: type of plot
    :type freq: float array
    :type H: float array
    :type freq_min: float
    :type freq_max: float
    :type FLAG: integer
    :returns: 

    =======  =============================================================
    FLAG     Plot Type
    -------  -------------------------------------------------------------
    1 (def)   Magnitude and Phase versus F
    2         Magnitude and Phase versus log10(F)
    3         Bodelog  (Magnitude and Phase versus log10(w))
    4         Real and Imaginary
    5         Nyquist  (Real versus Imaginary)
    6         Magnitude versus F 
    7         Phase versus F
    8         Real versus F
    9         Imaginary versus F
    10         Magnitude versus log10(F) 
    11         Phase versus log10(F)
    12         Real versus log10(F)
    13         Imaginary versus log10(F)
    14         Magnitude versus log10(w) 
    15         Phase versus log10(w)
    =======  =============================================================   
 

    :Example:

    >>> f=(0:.01:100)';
    >>> w=f*2*pi;
    >>> k=1e5;m=1;c=1;
    >>> tf=1./(m*(w*j).^2+c*j*w+k);
    >>> figure(1);frfplot(f,tf)
    >>> figure(2);frfplot(f,tf,5)

    Copyright J. Slater, Dec 17, 1994
    Updated April 27, 1995
    Ported to Python, July 1, 2015
    """
    freq=freq.reshape(1,-1)
    lenF=freq.shape[0]

    #    if lenF==1;
    #    F=(0:length(Xfer)-1)'*F;
    #    end
    if freq_max==0:
        freq_max=np.max(freq)
        #print(str(freq_max))
        
    if freq_min>freq_max:
        raise ValueError('freq_min must be less than freq_max.')

    #print(str(np.amin(freq)))
    inlow=lenF*(freq_min-np.amin(freq))//(np.amax(freq)-np.amin(freq))

    inhigh=lenF*(freq_max-np.amin(freq))//(np.amax(freq)-np.amin(freq))-1
    #if inlow<1,inlow=1;end
    #if inhigh>lenF,inhigh=lenF;end
    print(H.shape)
    H=H[:,inlow:inhigh]
    #print(H.shape)
    freq=freq[:,inlow:inhigh]
    mag=20*np.log10(np.abs(H))
    
    minmag=np.amin(mag)
    maxmag=np.amax(mag)
    phase=np.unwrap(np.angle(H))*180/np.pi;
    #    phmin_max=[min(phase)//45)*45 ceil(max(max(phase))/45)*45];
    phmin=np.amin(phase)//45*45.0
    phmax=(np.amax(phase)//45+1)*45
    minreal=np.amin(np.real(H))
    maxreal=np.amax(np.real(H))
    minimag=np.amin(np.imag(H))
    maximag=np.amax(np.imag(H))
    if FLAG==1:
        plt.subplot(2,1,1)
        plt.plot(freq.T,mag.T)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Mag (dB)')
        plt.grid()
        plt.xlim(xmax=freq_max,xmin=freq_min)
        plt.ylim(ymax=maxmag, ymin=minmag)
        
        plt.subplot(2,1,2)
        plt.plot(freq.T, phase.T)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (deg)')
        plt.grid()
        plt.xlim(xmax=freq_max,xmin=freq_min)
        plt.ylim(ymax=phmax, ymin=phmin)

        plt.yticks(np.arange(phmin,(phmax+45),45))
        plt.show()

    ##  elif FLAG==2:
    ##   subplot(2,1,1)
    ##   semilogx(F,mag)
    ##   xlabel('Frequency (Hz)')
    ##   ylabel('Mag (dB)')
    ##   grid on
    ## %  Fmin,Fmax,min(mag),max(mag)
    ##   axis([Fmin Fmax minmag maxmag])

    ##   subplot(2,1,2)
    ##   semilogx(F,phase)
    ##   xlabel('Frequency (Hz)')
    ##   ylabel('Phase (deg)')
    ##   grid on
    ##   axis([Fmin Fmax  phmin_max(1) phmin_max(2)])
    ##   gridmin_max=round(phmin_max/90)*90;
    ##   set(gca,'YTick',gridmin_max(1):90:gridmin_max(2))

    ##  elif FLAG==3:
      ## subplot(2,1,1)
      ## mag=20*log10(abs(Xfer));
      ## semilogx(F*2*pi,mag)
      ## xlabel('Frequency (Rad/s)')
      ## ylabel('Mag (dB)')
      ## grid on
      ## axis([Wmin Wmax minmag maxmag])
      ## zoom on
      ## subplot(2,1,2)
      ## semilogx(F*2*pi,phase)
      ## xlabel('Frequency (Rad/s)')
      ## ylabel('Phase (deg)')
      ## grid on
      ## axis([Wmin Wmax  phmin_max(1) phmin_max(2)])
      ## gridmin_max=round(phmin_max/90)*90;
      ## set(gca,'YTick',gridmin_max(1):90:gridmin_max(2))

     ## elseif FLAG==4
     ##  subplot(2,1,1)
     ##  plot(F,real(Xfer))
     ##  xlabel('Frequency (Hz)')
     ##  ylabel('Real')
     ##  grid on
     ##  axis([Fmin Fmax minreal maxreal])
     ##  zoom on
     ##  subplot(2,1,2)
     ##  plot(F,imag(Xfer))
     ##  xlabel('Frequency (Hz)')
     ##  ylabel('Imaginary')
     ##  grid on
     ##  axis([Fmin Fmax minimag maximag])
     ##  zoom on
     ## elseif FLAG==5
     ##  subplot(1,1,1)
     ##  imax=round(length(F)*Fmax/max(F));
     ##  imin=round(length(F)*Fmin/max(F))+1;
     ##  plot(real(Xfer(imin:imax)),imag(Xfer(imin:imax)))
     ##  xlabel('Real')
     ##  ylabel('Imaginary')
     ##  grid on
     ##  zoom on
     ## elseif FLAG==6
     ##  subplot(1,1,1)
     ##  mag=20*log10(abs(Xfer));
     ##  plot(F,mag)
     ##  xlabel('Frequency (Hz)')
     ##  ylabel('Mag (dB)')
     ##  grid on
     ##  axis([Fmin Fmax minmag maxmag])
     ##  zoom on
     ## elseif FLAG==7
     ##  subplot(1,1,1)
     ##  plot(F,phase)
     ##  xlabel('Frequency (Hz)')
     ##  ylabel('Phase (deg)')
     ##  grid on
     ##  phmin_max=[floor(min(phase)/45)*45 ceil(max(phase)/45)*45];
     ##  axis([Fmin Fmax  phmin_max(1) phmin_max(2)])
     ##  gridmin_max=round(phmin_max/90)*90;
     ##  set(gca,'YTick',gridmin_max(1):90:gridmin_max(2))
     ##  zoom on
     ## elseif FLAG==8
     ##  subplot(1,1,1)
     ##  plot(F,real(Xfer))
     ##  xlabel('Frequency (Hz)')
     ##  ylabel('Real')
     ##  grid on
     ##  axis([Fmin Fmax minreal maxreal])
     ##  zoom on
     ## elseif FLAG==9
     ##  subplot(1,1,1)
     ##  plot(F,imag(Xfer))
     ##  xlabel('Frequency (Hz)')
     ##  ylabel('Imaginary')
     ##  grid on
     ##  axis([Fmin Fmax minimag maximag])
     ##  zoom on
     ## elseif FLAG==10
     ##  subplot(1,1,1)
     ##  mag=20*log10(abs(Xfer));
     ##  semilogx(F,mag)
     ##  xlabel('Frequency (Hz)')
     ##  ylabel('Mag (dB)')
     ##  grid on
     ##  axis([Fmin Fmax minmag maxmag])
     ##  zoom on
     ## elseif FLAG==11
     ##  subplot(1,1,1)
     ##  semilogx(F,phase)
     ##  xlabel('Frequency (Hz)')
     ##  ylabel('Phase (deg)')
     ##  grid on
     ##  phmin_max=[floor(min(phase)/45)*45 ceil(max(phase)/45)*45];
     ##  axis([Fmin Fmax  phmin_max(1) phmin_max(2)])
     ##  gridmin_max=round(phmin_max/90)*90;
     ##  set(gca,'YTick',gridmin_max(1):90:gridmin_max(2))
     ##  zoom on
     ## elseif FLAG==12
     ##  subplot(1,1,1)
     ##  semilogx(F,real(Xfer))
     ##  xlabel('Frequency (Hz)')
     ##  ylabel('Real')
     ##  grid on
     ##  axis([Fmin Fmax minreal maxreal])
     ##  zoom on
     ## elseif FLAG==13
     ##  subplot(1,1,1)
     ##  semilogx(F,imag(Xfer))
     ##  xlabel('Frequency (Hz)')
     ##  ylabel('Imaginary')
     ##  grid on
     ##  axis([Fmin Fmax minimag maximag])
     ##  zoom on
     ## elseif FLAG==14
     ##  subplot(1,1,1)
     ##  mag=20*log10(abs(Xfer));
     ##  semilogx(F*2*pi,mag)
     ##  xlabel('Frequency (Rad/s)')
     ##  ylabel('Mag (dB)')
     ##  grid on
     ##  axis([Wmin Wmax minmag maxmag])
     ##  zoom on
     ## elseif FLAG==15
     ##  subplot(1,1,1)
     ##  semilogx(F*2*pi,phase)
     ##  xlabel('Frequency (Rad/s)')
     ##  ylabel('Phase (deg)')
     ##  grid on
     ##  axis([Wmin Wmax  phmin_max(1) phmin_max(2)])
     ##  gridmin_max=round(phmin_max/90)*90;
     ##  set(gca,'YTick',gridmin_max(1):90:gridmin_max(2))
     ##  zoom on
     ## else
     ##  subplot(2,1,1)
     ##  mag=20*log10(abs(Xfer));
     ##  plot(F,mag)
     ##  xlabel('Frequency (Hz)')
     ##  ylabel('Mag (dB)')
     ##  grid on
     ##  axis([Fmin Fmax minmag maxmag])
     ##  zoom on
     ##  subplot(2,1,2)
     ##  plot(F,phase)
     ##  xlabel('Frequency (Hz)')
     ##  ylabel('Phase (deg)')
     ##  grid on
     ##  phmin_max=[floor(min(phase)/45)*45 ceil(max(phase)/45)*45];
     ##  axis([Fmin Fmax phmin_max(1) phmin_max(2)])
     ##  gridmin_max=round(phmin_max/90)*90;
     ##  set(gca,'YTick',gridmin_max(1):90:gridmin_max(2))
     ##  zoom on
    
def xcorr(t, x, y, zeropad = True):

    tau = t
    sx = len(x)
    sy = len(y)
    if zeropad == True:
        Xn = sp.fft(x, n = len(x)*2)
        Yn = sp.conj(sp.fft(y, n = len(x)*2))
    else:
        Xn = sp.fft(x)
        Yn = sp.conj(sp.fft(y))

    xcor = sp.real(fftpack.fftshift(sp.ifft(Xn*Yn)))
    dt = t[1]-t[0]
    
    tau = sp.linspace(-len(xcor)/2*dt-dt/2,len(xcor)/2*dt-dt/2,len(xcor))
    return tau, xcor

    
'''
    function [tout,crcorout]=crcor(x,y,dt,type,ave)
%CRCOR Cross correlation.
% [Tau,COR]=CRCOR(X,Y,DT,TYPE,AVE) returns the Cross Correlation 
% between signals X and Y.
% [Tau,COR]=CRCOR(X,X,DT,TYPE,AVE) returns the Auto Correlation 
% of the signal X.
% DT is the time between samples.
% If DT is the time vector, DT is extracted as T(2)-T(1).
% TYPE is the type of correlation. TYPE = 1 causes CRCOR
% to return the linear correlation function. TYPE = 2
% causes CRCOR to return the circular correlation function.
% The default value is 1.
% If X and Y are matrices, averaging will be performed on the
% Correlations unless AVE is set to 'noave'. TYPE and AVE are 
% optional. Either can be left out.
%
% COH(X,Y,DT,N,AVE) plots the Correlation if there are no ouput 
% arguments. Click in the region of interest to zoom in. 
% Each click will double the size of the plot. Double click 
% to return to full scale.
%
% See also TFEST, ASD, COH, CRSD, and TFPLOT.

%	Copyright (c) 1994 by Joseph C. Slater
sy=size(y);
sy=size(y);
if nargin==3
  type=1;
  ave='yes';
 elseif nargin==4
  if strcmp(type,'noave')
   ave=n;
   type=1;
  else
   ave='yes';
  end
end

if isempty(type)
  type=1;
end

sx=size(x);
nc=sx(2);

if type==1
  n=sx(1)*2;
 else
  n=sx(1);
end

if length(dt)~=1
 dt=dt(2)-dt(1);
end

tmax=dt*(length(x)-1);
t=(-tmax:(2*tmax/(n-1)):tmax)'-(tmax/(n-1));

X=fft(x,n);
Y=fft(y,n);
pxy=real(ifft(conj(X).*Y));

crcr=fftshift(real(pxy));
crcr=crcr(1:length(crcr),:);

if nc~=1 & ~strcmp(ave,'noave')
 crcr=mean(crcr')';
end

if nargout==0
 plot(t,crcr)
 %logo
 if type==1
   text1='Linear ';
  else
   text1='Circular ';
 end
 if x==y
   text2='Auto ';
  else
   text2='Cross ';
 end
 text3=[text1 text2 'Correlation'];
 title(text3)
 xlabel('Time')
 ylabel(text3)
 grid
 zoom on
 return
end

crcorout=crcr;
tout=t;
'''
    
if __name__ == "__main__":
    import doctest
    #import vibrationtesting as vt
    #doctest.testmod(optionflags=doctest.ELLIPSIS)
    doctest.run_docstring_examples(frfest,globals(),optionflags=doctest.ELLIPSIS)
    #doctest.run_docstring_examples(asd,globals(),optionflags=doctest.ELLIPSIS)
    """ What this does. 
    python (name of this file)  -v
    will test all of the examples in the help.

    Leaving off -v will run the tests without any output. Success will return nothing.

    See the doctest section of the Sphinx manual.
    """
