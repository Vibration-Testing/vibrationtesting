"""
Signal processing, creation and plotting.

Analysis of data and generation of simulated experiments.
"""
__license__ = "Joseph C. Slater"

__docformat__ = 'reStructuredText'

# import warnings

import numpy as np
import scipy as sp
import scipy.fftpack as fftpack
import scipy.linalg as la
import matplotlib.pyplot as plt
import scipy.integrate as spi
import scipy.signal as signal

"""
Notes:
------
Sept. 3, 2016
Development of windows in scipy.signal has been rapid and
determining what I should build into this module, or simply leverage
from scipy.signal has been a moving target.
It's now apparent that creating or returning a window is pointless. Further,
Applying should be a relatively simple code obviating much of any need for the
code here.

The cross spectrum analysis formerly lacking is now available, periodogram
is usually the best option, however not with impulse excitations. See
`scipy.signal` for this. Unfortunately, the conventions in this module are not
consistent with `scipy.signal`. They follow those of `python-control`

FRF calculation is typically trivial, Hv being an expected gap long term
MIMO FRF calculation is an open question. Pretty printing of FRFs is always
a welcome tool.

System ID is likely the remaining missing aspect at this time.

In order to be consistent with the Control Systems Library, increasing time
or increasing frequency steps positively with increased column number (one
dimension). Rows (0 dimension)
correspond to appropriate channels, output numbers, etc.

For cross spectrum data (cross spectrum density, frequency response function)
the 2 dimension represents the input channel.

The last dimension (2 or 3) indexes each data instance (experiment). That means
that an unaveraged cross spectrum density has dimension 4. If there is only a
single input channel, it is imperative to insist the dimention exist, even if
only length 1. This is analagous to a vector being Nx1 versus simply a
1-D array of length 1.

http://python-control.readthedocs.io/en/latest/conventions.html#time-series-data

Problem: This hasn't been fully implemented.
"""


def window(x, windowname='hanning', normalize=False):
    r"""Create leakage window.

    Create a  window of length :math:`x`, or a window sized to match
    :math:`x` that :math:`x\times w` is the windowed result.

    Parameters
    ----------
    x: integer, float array
       | If integer- number of points in desired hanning windows.
       | If array- array provides size of window returned.
    windowname: string
       One of: hanning, hamming, blackman, flatwin, boxwin
    normalize: bool, optional(False)
       Adjust power level (for use in ASD) to 1

    Returns
    -------
    w: float array
       | window array of size x
       | window array. Windowed array is then :math:`x\times w`

    Examples
    --------
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
    >>> xcos = A*np.cos(2*np.pi*freq*time) # assembling individual records.
    >>> x=np.dstack((xsin,xcos)) # assembling individual records. vstack
    >>> xw=vt.hanning(x)*x
    >>> fig, (ax1, ax2) = plt.subplots(2,1)
    >>> ax1.plot(time.T,x[:,:,1].T)
    [<matplotlib.lines.Line2D object at ...>]
    >>> ax1.set_ylim([-20, 20])
    (-20, 20)
    >>> ax1.set_title('Original (raw) data.')
    Text(0.5,1,'Original (raw) data.')
    >>> ax1.set_ylabel('$x(t)$')
    Text(0,0.5,'$x(t)$')
    >>> ax2.plot(time[0,:],xw[0,:],time[0,:],vt.hanning(x)[0,:]*A,'--',
    ...                            time[0,:],-vt.hanning(x)[0,:]*A,'--')
    [<matplotlib.lines.Line2D object at ...>]
    >>> ax2.set_ylabel('Hanning windowed $x(t)$')
    Text(0,0.5,'Hanning windowed $x(t)$')
    >>> ax2.set_xlabel('time')
    Text(0.5,0,'time')
    >>> ax2.set_title('Effect of window. Note the scaling to conserve ASD amplitude')
    Text(0.5,1,'Effect of window. Note the scaling to conserve ASD amplitude')
    >>> fig.tight_layout()

    """
    if isinstance(x, (list, tuple, np.ndarray)):
        """Create Hanning windowing array of dimension `n` by `N` by `nr`
        where `N` is number of data points and `n` is the number of number of
        inputs or outputs and `nr` is the number of records."""

        swap = 0
        if len(x.shape) == 1:
            # We have either a scalar or 1D array
            if x.shape[0] == 1:
                print("x is a scalar... and shouldn\'t have entered this \
                      part of the loop.")
            else:
                N = len(x)

            f = window(N, windowname=windowname)

        elif len(x.shape) == 3:

            if x.shape[0] > x.shape[1]:
                x = np.swapaxes(x, 0, 1)
                swap = 1
                print('You shouldn\'t do that.')
                print('The 1 dimension is the time (or frequency) \
                       incrementing dimension.')
                print('Swapping axes temporarily to be compliant with \
                      expectations. I\'ll fix them in your result')

            N = x.shape[1]
            f = window(N, windowname=windowname)
            f, _, _ = np.meshgrid(f, np.arange(
                x.shape[0]), np.arange(x.shape[2]))
            if swap == 1:
                f = np.swapaxes(f, 0, 1)

        elif len(x.shape) == 2:

            if x.shape[0] > x.shape[1]:
                x = np.swapaxes(x, 0, 1)
                swap = 1
                print('You shouldn\'t do that.')
                print('The 1 dimension is the time (or frequency) ' +
                      'incrementing dimension.')
                print('Swapping axes temporarily to be compliant with ' +
                      'expectations.')
                print('I\'ll reluctantly return a transposed result.')

            f = window(x.shape[1], windowname=windowname)
            f, _ = np.meshgrid(f, np.arange(x.shape[0]))
            if swap == 1:
                f = np.swapaxes(f, 0, 1)

    else:
        N = x
        if windowname is 'hanning':
            f = np.sin(np.pi * np.arange(N) / (N - 1))**2 * np.sqrt(8 / 3)
        elif windowname is 'hamming':
            f = (0.54 - 0.46 * np.cos(2 * np.pi * (np.arange(N)) / (N - 1)))\
                * np.sqrt(5000 / 1987)
        elif windowname is 'blackman':
            print('blackman')
            f = (0.42 - 0.5 * np.cos(2 * np.pi * (np.arange(N) + .5) / (N))
                 + .08 * np.cos(4 * np.pi * (np.arange(N) + .5) / (N)))\
                * np.sqrt(5000 / 1523)
        elif windowname is 'flatwin':
            f = 1.0 - 1.933 * np.cos(2 * np.pi * (np.arange(N)) / (N - 1))\
                + 1.286 * np.cos(4 * np.pi * (np.arange(N)) / (N - 1))\
                - 0.338 * np.cos(6 * np.pi * (np.arange(N)) / (N - 1))\
                + 0.032 * np.cos(8 * np.pi * (np.arange(N)) / (N - 1))
        elif windowname is 'boxwin':
            f = np.ones((1, N))
        else:
            f = np.ones((1, N))
            print("I don't recognize window name ", windowname, ". Sorry.")

        if normalize is True:
            f = f / la.norm(f) * np.sqrt(N)
    return f


def hanning(x, normalize=False):
    r"""Return hanning window.

    Create a hanning window of length :math:`x`, or a hanning window sized to
    match :math:`x` that :math:`x\times w` is the windowed result.

    Parameters
    ----------
    x: integer, float array
       | If integer- number of points in desired hanning windows.
       | If array- array provides size of window returned.
    windowname: string
       One of: hanning, hamming, blackman, flatwin, boxwin
    normalize: bool, optional(False)
       Adjust power level (for use in ASD) to 1

    Returns
    -------
    w: float array
       | window array of size x
       | window array. Windowed array is then :math:`x\times w`

    Examples
    --------
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
    >>> x=np.dstack((xsin,xcos)) # assembling individual records. vstack
    >>> xw=vt.hanning(x)*x
    >>> fig, (ax1, ax2) = plt.subplots(2, 1)
    >>> ax1.plot(time.T,x[:,:,1].T)
    [<matplotlib.lines.Line2D object at ...>]
    >>> ax1.set_ylim([-20, 20])
    (-20, 20)
    >>> ax1.set_title('Unwindowed data, 2 records.')
    Text(0.5,1,'Unwindowed data, 2 records.')
    >>> ax1.set_ylabel('$x(t)$')
    Text(0,0.5,'$x(t)$')
    >>> ax2.plot(time[0,:],xw[0,:],time[0,:],vt.hanning(x)[0,:]*A,
    ...                      '--',time[0,:],-vt.hanning(x)[0,:]*A,'--')
    [<matplotlib.lines.Line2D object at ...>]
    >>> ax2.set_ylabel('Hanning windowed $x(t)$')
    Text(0,0.5,'Hanning windowed $x(t)$')
    >>> ax2.set_xlabel('time')
    Text(0.5,0,'time')
    >>> ax2.set_title('Effect of window. Note the scaling to conserve ASD amplitude')
    Text(0.5,1,'Effect of window. Note the scaling to conserve ASD amplitude')
    >>> fig.tight_layout()

    """
    if isinstance(x, (list, tuple, np.ndarray)):
        """Create Hanning windowing array of dimension n by N by nr
        where N is number of data points and n is the number of number of
        inputs or outputs and nr is the number of records."""

        swap = 0
        if len(x.shape) == 1:
            # We have either a scalar or 1D array
            if x.shape[0] == 1:
                print("x is a scalar... and shouldn\'t have \
                       entered this part of the loop.")
            else:
                N = len(x)
            f = hanning(N)

        elif len(x.shape) == 3:
            # print('a')
            # print(f.shape)

            if x.shape[0] > x.shape[1]:
                x = np.swapaxes(x, 0, 1)
                swap = 1
                print('Swapping axes temporarily to be compliant with \
                      expectations. I\'ll fix them in your result')

            f = hanning(x.shape[1])
            f, _, _ = np.meshgrid(f, np.arange(
                x.shape[0]), np.arange(x.shape[2]))
            if swap == 1:
                f = np.swapaxes(f, 0, 1)

        elif len(x.shape) == 2:
            # f,_=np.meshgrid(f[0,:],np.arange(x.shape[0]))
            # print('b')
            # print('length = 2')
            # print(x.shape)
            if x.shape[0] > x.shape[1]:
                x = np.swapaxes(x, 0, 1)
                swap = 1
                print('Swapping axes temporarily to be compliant with \
                      expectations. I\'ll fix them in your result')
            f = hanning(x.shape[1])
            f, _ = np.meshgrid(f, np.arange(x.shape[0]))
            if swap == 1:
                f = np.swapaxes(f, 0, 1)
    else:
        # print(x)
        # Create hanning window of length x
        N = x
        # print(N)
        f = np.sin(np.pi * np.arange(N) / (N - 1))**2 * np.sqrt(8 / 3)
        if normalize is True:
            f = f / la.norm(f) * np.sqrt(N)
    return f


def blackwin(x):
    """Return the n point Blackman window.

    Returns x as the Blackman windowing array x_window
    The windowed signal is then x*x_window
    """
    print('blackwin is untested')
    if isinstance(x, (list, tuple, np.ndarray)):
        n = x.shape[1]
        f = blackwin(n)

        if len(x.shape) == 3:
            f, _, _ = np.meshgrid(f[0, :], np.arange(
                x.shape[0]), np.arange(x.shape[2]))
        else:
            f, _ = np.meshgrid(f[0, :], np.arange(x.shape[0]))
    else:
        n = x
        f = np.reshape((0.42 - 0.5 * np.cos(2 * np.pi * (np.arange(n) + .5)) /
                        (n) + .08 * np.cos(4 * np.pi * (np.arange(n) + .5)) /
                        (n)) * np.sqrt(5000 / 1523), (1, -1))
        f = f / la.norm(f) * np.sqrt(n)
    return f


def expwin(x, ts=.75):
    """Return the n point exponential window.

    Returns x as the expwin windowing array x_windowed
    The windowed signal is then x*x_window
    The optional second argument set the 5% "settling time" of the window.
    Default is ts=0.75
    """
    print('expwin is untested')
    tc = -ts / np.log(.05)
    if isinstance(x, (list, tuple, np.ndarray)):
        n = x.shape[1]
        f = expwin(n)

        if len(x.shape) == 3:
            f, _, _ = np.meshgrid(f[0, :], np.arange(
                x.shape[0]), np.arange(x.shape[2]))
        else:
            f, _ = np.meshgrid(f[0, :], np.arange(x.shape[0]))
    else:
        n = x
        v = (n - 1) / n * np.arange(n) + (n - 1) / n / 2
        f = np.exp(-v / tc / (n - 1))
        f = f / la.norm(f) * np.sqrt(n)
        f = np.reshape(f, (1, -1))
        f = f / la.norm(f) * np.sqrt(n)

    return f


def hammwin(x):
    """Return the n point hamming window.

    Returns x as the hamming windowingarray x_windowed
    The windowed signal is then x*x_window
    """
    print('hammwin is untested')
    if isinstance(x, (list, tuple, np.ndarray)):
        n = x.shape[1]
        f = hammwin(n)

        if len(x.shape) == 3:
            f, _, _ = np.meshgrid(f[0, :], np.arange(
                x.shape[0]), np.arange(x.shape[2]))
        else:
            f, _ = np.meshgrid(f[0, :], np.arange(x.shape[0]))
    else:

        n = x
        f = np.reshape((0.54 - 0.46 * np.cos(2 * np.pi * (np.arange(n)) /
                                             (n - 1))) * np.sqrt(5000 / 1987),
                       (1, -1))
        f = f / la.norm(f) * np.sqrt(n)

    return f


def flatwin(x):
    """Return the n point flat top window.

    x_windows=flatwin(x)
    Returns x as the flat top windowing array x_windowed
    The windowed signal is then x*x_window
    McConnell, K. G., "Vibration Testing: Theory and Practice," Wiley, 1995.
    """
    print('flatwin is untested')
    if isinstance(x, (list, tuple, np.ndarray)):
        n = x.shape[1]
        f = flatwin(n)

        if len(x.shape) == 3:
            f, _, _ = np.meshgrid(f[0, :], np.arange(
                x.shape[0]), np.arange(x.shape[2]))
        else:
            f, _ = np.meshgrid(f[0, :], np.arange(x.shape[0]))
    else:

        n = x
        f = np.reshape(
            (1.0 - 1.933 * np.cos(2 * np.pi * (np.arange(n)) / (n - 1))
             + 1.286 * np.cos(4 * np.pi * (np.arange(n)) / (n - 1))
                 - 0.338 * np.cos(6 * np.pi * (np.arange(n)) / (n - 1))
             + 0.032 * np.cos(8 * np.pi * (np.arange(n)) / (n - 1))),
            (1, -1))
        f = f / la.norm(f) * np.sqrt(n)

    return f


def boxwin(x):
    """Return the n point box window (uniform).

    Returns x as the boxwin windowing array x_windowed
    The windowed signal is then x*x_window
    """
    print('boxwin is untested')
    if isinstance(x, (list, tuple, np.ndarray)):
        n = x.shape[1]
        f = boxwin(n)

        if len(x.shape) == 3:
            f, _, _ = np.meshgrid(f[0, :], np.arange(
                x.shape[0]), np.arange(x.shape[2]))
        else:
            f, _ = np.meshgrid(f[0, :], np.arange(x.shape[0]))
    else:

        n = x
        # f=np.reshape((1.0-1.933*np.cos(2*np.pi*(np.arange(n))/(n-1))+1.286*np.cos(4*np.pi*(np.arange(n))/(n-1))-0.338*np.cos(6*np.pi*(np.arange(n))/(n-1))+0.032*np.cos(8*np.pi*(np.arange(n))/(n-1))),(1,-1))
        f = np.reshape(np.ones((1, n)), (1, -1))
        f = f / la.norm(f) * np.sqrt(n)

    return f


def hannwin(*args, **kwargs):
    """Alternative for function `hanning`."""
    return hanning(*args, **kwargs)


def asd(x, t, windowname="none", ave=bool(True)):
    """Return autospectrum (power spectrum) density of a signal x.

    Parameters
    ----------
    x : float array
        Data array (n x N x m) where n is the number of sensors, m the
        number of experiments.
    t : float array
        Time array (1 x N)
    windowname : string
        Name of windowing function to use. See `window`.
    ave : bool, optional(True)
        Average result or not?

    Returns
    -------
    f : float array
        Frequency vector (1 x N)
    Pxx : float array
          Autospectrum (n x N) or (n x N x m) if not averaged.

    Examples
    --------
    >>> from scipy import signal
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import vibrationtesting as vt
    >>> import numpy.linalg as la

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
    >>> x = x + np.random.normal(scale=np.sqrt(noise_power),
    ...                              size=(1, time.shape[1]))
    >>> fig, (ax1, ax2) = plt.subplots(2,1)
    >>> ax1.plot(time[0,:],x[0,:])
    [<matplotlib.lines.Line2D object at ...>]
    >>> ax1.set_title('Time history')
    Text(0.5,1,'Time history')
    >>> ax1.set_xlabel('Time (sec)')
    Text(0.5,0,'Time (sec)')
    >>> ax1.set_ylabel('$x(t)$')
    Text(0,0.5,'$x(t)$')

    Compute and plot the autospectrum density.

    >>> freq_vec, Pxx = vt.asd(x, time, windowname="hanning", ave=bool(False))
    >>> ax2.plot(freq_vec, 20*np.log10(Pxx[0,:]))
    [<matplotlib.lines.Line2D object at ...>]
    >>> ax2.set_ylim([-400, 100])
    (-400, 100)
    >>> ax2.set_xlabel('frequency (Hz)')
    Text(0.5,0,'frequency (Hz)')
    >>> ax2.set_ylabel('PSD (V**2/Hz)')
    Text(0,0.5,'PSD (V**2/Hz)')

    If we average the last half of the spectral density, to exclude the
    peak, we can recover the noise power on the signal.

    """
    f, Pxx = crsd(x, x, t, windowname=windowname, ave=ave)
    Pxx = Pxx.real
    return f, Pxx


def crsd(x, y, t, windowname="none", ave=bool(True)):
    """
    Calculate the cross spectrum (power spectrum) density between two signals.

    Parameters
    ----------
    x, y : arrays
        Data array (n x N x m) where n is the number of sensors, m the
        number of experiments.
    t : array
        Time array (1 x N)
    windowname : string
        Name of windowing function to use. See `window`.
    ave : bool, optional
        Average result or not?

    Returns
    -------
    f : array
        Frequency vector (1 x N)
    Pxy : array
          Autospectrum (n x N) or (n x N x m) if not averaged.

    Examples
    --------
    >>> from scipy import signal
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import vibrationtesting as vt
    >>> import numpy.linalg as la

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
    >>> x = x + np.random.normal(scale=np.sqrt(noise_power),
    ...                          size=(1, time.shape[1]))
    >>> fig = plt.figure()
    >>> plt.subplot(2,1,1)
    <matplotlib...>
    >>> plt.plot(time[0,:],x[0,:])
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.title('Time history')
    Text(0.5,1,'Time history')
    >>> plt.xlabel('Time (sec)')
    Text(0.5,0,'Time (sec)')
    >>> plt.ylabel('$x(t)$')
    Text(0,0.5,'$x(t)$')

    Compute and plot the autospectrum density.
    >>> freq_vec, Pxx = vt.asd(x, time, windowname="hanning", ave=bool(False))
    >>> plt.subplot(2,1,2)
    <matplotlib...>
    >>> plt.plot(freq_vec, 20*np.log10(Pxx[0,:]))
    [<matplotlib.lines.Line2D object at ...>]
    >>> plt.ylim([-400, 100])
    (-400, 100)
    >>> plt.xlabel('frequency (Hz)')
    Text(0.5,0,'frequency (Hz)')
    >>> plt.ylabel('PSD (V**2/Hz)')
    Text(0,0.5,'PSD (V**2/Hz)')
    >>> fig.tight_layout()

    """
    # t_shape = t.shape
    t = t.flatten()
    if len(t) == 1:
        dt = t
    else:
        dt = t[2] - t[1]

    if dt <= 0:
        print('You sent in bad data. Delta t is negative. \
              Please check your inputs.')

    if len(x.shape) == 1:
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=2)
        y = np.expand_dims(y, axis=0)
        y = np.expand_dims(y, axis=2)
    n = x.shape[1]

    if windowname is False or windowname.lower() is "none":
        win = 1
    else:
        # print('This doesn\'t work yet')
        windowname = windowname.lower()
        win = 1
        if windowname == "hanning":
            win = window(x, windowname='hanning')
        elif windowname == "blackwin":
            win = window(x, windowname='blackwin')
        elif windowname == "boxwin":
            win = window(x, windowname='boxwin')
        elif windowname == "expwin":
            win = window(x, windowname='expwin')
        elif windowname == "hammwin":
            win = window(x, windowname='hamming')
        elif windowname == "triwin":
            win = window(x, windowname='triwin')
        elif windowname == "flatwin":
            win = window(x, windowname='flatwin')

        y = y * win
        x = x * win
        del win

    ffty = np.fft.rfft(y, n, axis=1) * dt
    fftx = np.fft.rfft(x, n, axis=1) * dt

    Pxy = np.conj(fftx) * ffty / (n * dt) * 2

    if len(Pxy.shape) == 3 and Pxy.shape[2] > 1 and ave:
        Pxy = np.mean(Pxy, 2)

    nfreq = 1 / dt / 2
    f = np.linspace(0, nfreq, Pxy.shape[1])  # /2./np.pi

    return f, Pxy


def frfest(x, f, dt, windowname="hanning", ave=bool(True), Hv=bool(False)):
    r"""Return freq, H1, H2, coh, Hv.

    Estimates the :math:`H(j\omega)` Frequency Response Functions (FRFs)
    between :math:`x` and :math:`f`.

    Parameters
    ----------
    x : float array
        output or response of system
    f : float array
        input to system
    dt : float
        time step of samples
    windowname : string
        One of: hanning, hamming, blackman, flatwin, boxwin
    ave : bool, optional(True)- currently locked
        whether or not to average PSDs and ASDs or calculate raw FRFs
    Hv : bool, optional(False)
        calculate the :math:`H_v` frequency response function

    Returns
    -------
    freq : float array
        frequency vector (1xN)
    H1 :  float array
        Frequency Response Function :math:`H_1` estimate, (nxN) or (nxNxm)
    H2 :  float array
        Frequency Response Function :math:`H_2` estimate, (nxN) or (nxNxm)
    coh :  float array
        Coherance Function :math:`\gamma^2` estimate, (nxN)
    Hv : float array
        Frequency Response Function :math:`H_v` estimate, (nxN) or (nxNxm)

    Currently  ``ave`` is locked to default values.

    Examples
    --------
    >>> import control as ctrl
    >>> import matplotlib.pyplot as plt
    >>> import vibrationtesting as vt
    >>> import numpy as np
    >>> sample_freq = 1e3
    >>> noise_power = 0.001 * sample_freq / 2
    >>> A = np.array([[0, 0, 1, 0],
    ...               [0, 0, 0, 1],
    ...               [-200, 100, -.2, .1],
    ...               [100, -200, .1, -.2]])
    >>> B = np.array([[0], [0], [1], [0]])
    >>> C = np.array([[35, 0, 0, 0], [0, 35, 0, 0]])
    >>> D = np.array([[0], [0]])
    >>> sys = ctrl.ss(A, B, C, D)
    >>> tin = np.arange(0, 51.2, .1)
    >>> nr = .5   # 0 is all noise on input
    >>> for i in np.arange(520):
    ...     u = np.random.normal(scale=np.sqrt(noise_power), size=tin.shape)
    ...     #print(u)
    ...     t, yout, xout = ctrl.forced_response(sys, tin, u,rtol=1e-12)
    ...     if 'Yout' in locals():
    ...         Yout=np.dstack((Yout,yout
    ...                 +nr*np.random.normal(scale=.050*np.std(yout[0,:]),
    ...                  size=yout.shape)))
    ...         Ucomb=np.dstack((Ucomb,u+(1-nr)
    ...                 *np.random.normal(scale=.05*np.std(u),
    ...                  size=u.shape)))
    ...     else:
    ...         Yout=yout+nr*np.random.normal(scale=.05*np.std(yout[0,:]),
    ...                   size=yout.shape)
    ...                   # noise on output is 5% scale of input
    ...         Ucomb=u+(1-nr)*np.random.normal(scale=.05*np.std(u),
    ...                   size=u.shape)#(1, len(tin)))
    ...                   # 5% noise signal on input
    >>> f, Hxy1, Hxy2, coh, Hxyv = vt.frfest(Yout, Ucomb, t, Hv=bool(True))
    >>> vt.frfplot(f,Hxy2,freq_max=3.5, legend=['$H_{11}$', '$H_{12}$'])
    ...               # doctest: +SKIP
    >>> vt.frfplot(f, np.vstack((Hxy1[0,:], Hxy2[0,:], Hxyv[0,:])),
    ...               legend=['$H_{11-1}$','$H_{11-2}$','$H_{11-v}$'])
    ...               # doctest: +SKIP

    Notes
    -----
    .. note:: Not compatible with scipy.signal functions
    .. seealso:: :func:`asd`, :func:`crsd`, :func:`frfplot`.
    .. warning:: hanning window cannot be selected yet. Averaging cannot be
       unslected yet.
    .. todo:: Fix averaging, windowing, multiple input.

    """
    if len(f.shape) == 1:
        f = f.reshape(1, -1, 1)

    if len(x.shape) == 1:
        x = x.reshape(1, -1, 1)

    if len(f.shape) == 2:
        if (f.shape).index(max(f.shape)) == 0:
            f = f.reshape(max(f.shape), min(f.shape), 1)
        else:
            f = f.reshape(1, max(f.shape), min(f.shape))

    if len(x.shape) == 2:
        if (x.shape).index(max(x.shape)) == 0:
            x = x.reshape(max(x.shape), min(x.shape), 1)
        else:
            x = x.reshape(1, max(x.shape), min(x.shape))

    # Note: Two different ways to ignore returned values shown
    Pff = asd(f, dt, windowname=windowname)[1]
    freq, Pxf = crsd(x, f, dt, windowname=windowname)
    _, Pxx = asd(x, dt)

    # Note Pfx=conj(Pxf) is applied in the H1 FRF estimation
    Txf1 = np.conj(Pxf / Pff)
    Txf2 = Pxx / Pxf
    # Nulled to avoid output problems/simplify calls if unrequested
    Txfv = np.zeros_like(Txf1)

    coh = (Pxf * np.conj(Pxf)).real / Pxx / Pff

    if Hv:
        for i in np.arange(Pxx.shape[1]):
            frfm = np.array(
                [[Pff[0, i], np.conj(Pxf[0, i])], [Pxf[0, i], Pxx[0, i]]])

            alpha = 1  # np.sqrt(Pff[0,i]/Pxx[0,i])

            frfm = np.array([[Pff[0, i], alpha * np.conj(Pxf[0, i])],
                             [alpha * Pxf[0, i], alpha**2 * Pxx[0, i]]])

            lam, vecs = la.eigh(frfm)

            index = lam.argsort()

            lam = lam[index]

            vecs = vecs[:, index]

            Txfv[0, i] = -(vecs[0, 0] / vecs[1, 0]) / alpha

    return freq, Txf1, Txf2, coh, Txfv


def frfplot(freq, H, freq_min=0, freq_max=None, type=1, legend=[]):
    """Frequency Response function pretty plotting.

    Plots frequency response functions in a variety of formats

    Parameters
    ----------
    freq : float array
        Frequency vector (rad/sec), (1xN)
    H : float array
        Frequency response functions (nxN)
    freq_min : float, optional
        Low frequency for plot (default 0)
    freq_min : float, optional
        High frequency for plot (default max frequency)
    legend : string array
        Array of string for use in legend.
    type : int, optional
        Plot type. See notes.

    Returns
    -------
    ax : axis objects
        allows manipulation of plot parameters (xlabel, title...)

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import vibrationtesting as vt
    >>> import numpy as np
    >>> f=np.linspace(0,100,10000).reshape(-1,1);
    >>> w=f*2*np.pi;
    >>> k=1e5;m=1;c=1;
    >>> frf1=1./(m*(w*1j)**2+c*1j*w+k)
    >>> frf2=1./(m*(w*1j)**2+c*1j*w+k*3)
    >>> _ = vt.frfplot(f,np.hstack((frf1,frf2)), legend = ['FRF 1','FRF 2'])
    ...                                      # doctest: +SKIP

    Notes
    -----
    +---------+------------------------------------------------+
    |  type   |  Plot style                                    |
    +=========+================================================+
    | 1 (def) |  Magnitude and Phase versus F                  |
    +---------+------------------------------------------------+
    | 2       |  Magnitude and Phase versus log10(F)           |
    +---------+------------------------------------------------+
    | 3       |  Bodelog (Magnitude and Phase versus log10(w)) |
    +---------+------------------------------------------------+
    | 4       |  Real and Imaginary                            |
    +---------+------------------------------------------------+
    | 5       |  Nyquist (Imaginary versus Real)               |
    +---------+------------------------------------------------+
    | 6       |  Magnitude versus F                            |
    +---------+------------------------------------------------+
    | 7       |  Phase versus F                                |
    +---------+------------------------------------------------+
    | 8       |  Real versus F                                 |
    +---------+------------------------------------------------+
    | 9       |  Imaginary versus F                            |
    +---------+------------------------------------------------+
    | 10      |  Magnitude versus log10(F)                     |
    +---------+------------------------------------------------+
    | 11      |  Phase versus log10(F)                         |
    +---------+------------------------------------------------+
    | 12      |  Real versus log10(F)                          |
    +---------+------------------------------------------------+
    | 13      |  Imaginary versus log10(F)                     |
    +---------+------------------------------------------------+
    | 14      |  Magnitude versus log10(w)                     |
    +---------+------------------------------------------------+
    | 15      |  Phase versus log10(w)                         |
    +---------+------------------------------------------------+

    .. seealso:: `frfest`

    Copyright J. Slater, Dec 17, 1994
    Updated April 27, 1995
    Ported to Python, July 1, 2015

    """
    FLAG = type  # Plot type, should libe renamed throughout.
    freq = freq.reshape(1, -1)
    lenF = freq.shape[1]
    if len(H.shape) is 1:
        H = H.reshape(1, -1)

    if H.shape[0] > H.shape[1]:
        H = H.T

    if freq_max is None:
        freq_max = np.max(freq)

    if freq_min is None:
        freq_min = np.min(freq)

    if freq_min < np.min(freq):
        freq_min = np.min(freq)

    if freq_min > freq_max:
        raise ValueError('freq_min must be less than freq_max.')

    # print(str(np.amin(freq)))
    inlow = int(lenF * (freq_min - np.amin(freq)
                        ) // (np.amax(freq) - np.amin(freq)))

    inhigh = int(lenF * (freq_max - np.amin(freq)
                         ) // (np.amax(freq) - np.amin(freq)) - 1)
    # if inlow<1,inlow=1;end
    # if inhigh>lenF,inhigh=lenF;end
    """print('freq shape: {}'.format(freq.shape))
    print('H shape: {}'.format(H.shape))
    print('Index of low frequency: {}'.format(inlow))
    print('Index of high frequency: {}'.format(inhigh))"""
    H = H[:, inlow:inhigh]
    # print(H.shape)
    freq = freq[:, inlow:inhigh]
    mag = 20 * np.log10(np.abs(H))
    # print(mag)
    # print(mag.shape)
    minmag = np.min(mag)
    maxmag = np.max(mag)
    phase = np.unwrap(np.angle(H)) * 180 / np.pi
    #    phmin_max=[min(phase)//45)*45 ceil(max(max(phase))/45)*45];
    phmin = np.amin(phase) // 45 * 45.0
    phmax = (np.amax(phase) // 45 + 1) * 45
    """minreal = np.amin(np.real(H))
    maxreal = np.amax(np.real(H))
    minimag = np.amin(np.imag(H))
    maximag = np.amax(np.imag(H))"""

    if FLAG is 1:
        fig, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(freq.T, mag.T)
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Mag (dB)')
        ax1.grid()
        ax1.set_xlim(xmax=freq_max, xmin=freq_min)
        ax1.set_ylim(ymax=maxmag, ymin=minmag)

        ax2.plot(freq.T, phase.T)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Phase (deg)')
        ax2.grid()
        ax2.set_xlim(xmax=freq_max, xmin=freq_min)
        ax2.set_ylim(ymax=phmax, ymin=phmin)
        ax2.set_yticks(np.arange(phmin, (phmax + 45), 45))
        fig.tight_layout()

        if len(legend) > 0:
            plt.legend(legend)
        ax = (ax1, ax2)
    else:
        print("Sorry, that option isn't supported yet")
    return ax

    """# elif FLAG==2:
    # subplot(2,1,1)
    # semilogx(F,mag)
    #   xlabel('Frequency (Hz)')
    #   ylabel('Mag (dB)')
    # grid on
    # %  Fmin,Fmax,min(mag),max(mag)
    # axis([Fmin Fmax minmag maxmag])

    # subplot(2,1,2)
    # semilogx(F,phase)
    # xlabel('Frequency (Hz)')
    # ylabel('Phase (deg)')
    # grid on
    # axis([Fmin Fmax  phmin_max(1) phmin_max(2)])
    # gridmin_max=round(phmin_max/90)*90;
    # set(gca,'YTick',gridmin_max(1):90:gridmin_max(2))

    # elif FLAG==3:
    # subplot(2,1,1)
    # mag=20*log10(abs(Xfer));
    # semilogx(F*2*pi,mag)
    # xlabel('Frequency (Rad/s)')
      # ylabel('Mag (dB)')
      # grid on
      # axis([Wmin Wmax minmag maxmag])
      # zoom on
      # subplot(2,1,2)
      # semilogx(F*2*pi,phase)
      # xlabel('Frequency (Rad/s)')
      # ylabel('Phase (deg)')
      # grid on
      # axis([Wmin Wmax  phmin_max(1) phmin_max(2)])
      # gridmin_max=round(phmin_max/90)*90;
      # set(gca,'YTick',gridmin_max(1):90:gridmin_max(2))

     # elseif FLAG==4
     # subplot(2,1,1)
     # plot(F,real(Xfer))
     #  xlabel('Frequency (Hz)')
     # ylabel('Real')
     # grid on
     # axis([Fmin Fmax minreal maxreal])
     # zoom on
     # subplot(2,1,2)
     # plot(F,imag(Xfer))
     #  xlabel('Frequency (Hz)')
     # ylabel('Imaginary')
     # grid on
     # axis([Fmin Fmax minimag maximag])
     # zoom on
     # elseif FLAG==5
     # subplot(1,1,1)
     # imax=round(length(F)*Fmax/max(F));
     # imin=round(length(F)*Fmin/max(F))+1;
     # plot(real(Xfer(imin:imax)),imag(Xfer(imin:imax)))
     # xlabel('Real')
     # ylabel('Imaginary')
     # grid on
     # zoom on
     # elseif FLAG==6
     # subplot(1,1,1)
     # mag=20*log10(abs(Xfer));
     # plot(F,mag)
     #  xlabel('Frequency (Hz)')
     #  ylabel('Mag (dB)')
     # grid on
     # axis([Fmin Fmax minmag maxmag])
     # zoom on
     # elseif FLAG==7
     # subplot(1,1,1)
     # plot(F,phase)
     #  xlabel('Frequency (Hz)')
     #  ylabel('Phase (deg)')
     # grid on
     # phmin_max=[floor(min(phase)/45)*45 ceil(max(phase)/45)*45];
     # axis([Fmin Fmax  phmin_max(1) phmin_max(2)])
     # gridmin_max=round(phmin_max/90)*90;
     # set(gca,'YTick',gridmin_max(1):90:gridmin_max(2))
     # zoom on
     # elseif FLAG==8
     # subplot(1,1,1)
     # plot(F,real(Xfer))
     #  xlabel('Frequency (Hz)')
     # ylabel('Real')
     # grid on
     # axis([Fmin Fmax minreal maxreal])
     # zoom on
     # elseif FLAG==9
     # subplot(1,1,1)
     # plot(F,imag(Xfer))
     #  xlabel('Frequency (Hz)')
     # ylabel('Imaginary')
     # grid on
     # axis([Fmin Fmax minimag maximag])
     # zoom on
     # elseif FLAG==10
     # subplot(1,1,1)
     # mag=20*log10(abs(Xfer));
     # semilogx(F,mag)
     #  xlabel('Frequency (Hz)')
     #  ylabel('Mag (dB)')
     # grid on
     # axis([Fmin Fmax minmag maxmag])
     # zoom on
     # elseif FLAG==11
     # subplot(1,1,1)
     # semilogx(F,phase)
     #  xlabel('Frequency (Hz)')
     #  ylabel('Phase (deg)')
     # grid on
     # phmin_max=[floor(min(phase)/45)*45 ceil(max(phase)/45)*45];
     # axis([Fmin Fmax  phmin_max(1) phmin_max(2)])
     # gridmin_max=round(phmin_max/90)*90;
     # set(gca,'YTick',gridmin_max(1):90:gridmin_max(2))
     # zoom on
     # elseif FLAG==12
     # subplot(1,1,1)
     # semilogx(F,real(Xfer))
     #  xlabel('Frequency (Hz)')
     # ylabel('Real')
     # grid on
     # axis([Fmin Fmax minreal maxreal])
     # zoom on
     # elseif FLAG==13
     # subplot(1,1,1)
     # semilogx(F,imag(Xfer))
     #  xlabel('Frequency (Hz)')
     # ylabel('Imaginary')
     # grid on
     # axis([Fmin Fmax minimag maximag])
     # zoom on
     # elseif FLAG==14
     # subplot(1,1,1)
     # mag=20*log10(abs(Xfer));
     # semilogx(F*2*pi,mag)
     #  xlabel('Frequency (Rad/s)')
     #  ylabel('Mag (dB)')
     # grid on
     # axis([Wmin Wmax minmag maxmag])
     # zoom on
     # elseif FLAG==15
     # subplot(1,1,1)
     # semilogx(F*2*pi,phase)
     #  xlabel('Frequency (Rad/s)')
     #  ylabel('Phase (deg)')
     # grid on
     # axis([Wmin Wmax  phmin_max(1) phmin_max(2)])
     # gridmin_max=round(phmin_max/90)*90;
     # set(gca,'YTick',gridmin_max(1):90:gridmin_max(2))
     # zoom on
     # else
     # subplot(2,1,1)
     # mag=20*log10(abs(Xfer));
     # plot(F,mag)
     #  xlabel('Frequency (Hz)')
     #  ylabel('Mag (dB)')
     # grid on
     # axis([Fmin Fmax minmag maxmag])
     # zoom on
     # subplot(2,1,2)
     # plot(F,phase)
     #  xlabel('Frequency (Hz)')
     #  ylabel('Phase (deg)')
     # grid on
     # phmin_max=[floor(min(phase)/45)*45 ceil(max(phase)/45)*45];
     # axis([Fmin Fmax phmin_max(1) phmin_max(2)])
     # gridmin_max=round(phmin_max/90)*90;
     # set(gca,'YTick',gridmin_max(1):90:gridmin_max(2))
     # zoom on
     """


def xcorr(t, x, y, zeropad=True):
    """Sorry, no docs or tests yet."""
    tau = t
    # sx = len(x)
    # sy = len(y)
    if zeropad is True:
        Xn = np.fft.rfft(x, n=len(x) * 2)
        Yn = np.conj(sp.fft(y, n=len(x) * 2))
    else:
        Xn = np.fft.rfft(x)
        Yn = np.conj(np.fft.rfft(y))

    xcor = np.real(fftpack.fftshift(sp.ifft(Xn * Yn)))
    dt = t[1] - t[0]

    tau = np.linspace(-len(xcor) / 2 * dt - dt / 2,
                      len(xcor) / 2 * dt - dt / 2, len(xcor))
    return tau, xcor


def hammer_impulse(time, imp_time=None, imp_duration=None, doublehit=False,
                   dh_delta=None):
    """Generate simulated hammer hit (half sine).

    Parameters
    ----------
    time : float array
        1 x N time array. Suggest using `np.linspace(0,10,1000).reshape(1,-1)`
        for example
    imp_time : float (optional)
        Time of onset of impulse. Default is 0.1 time end time- which
        traditionally works well for impact testing
    imp_duration : float (optional)
        Duration of impulse. Default is 0.01 of total record
    doublehit : Boolean (optional)
        Allows repeat of hit to emulate a bad strike. Default is False
    dh_delta : float (optional)
        Time difference between primary strike and accidental second strike
        Default is 0.02 of record.

    Returns
    -------
    force : float array

    Examples
    --------
    >>> import vibrationtesting as vt
    >>> time = np.linspace(0,10,1024).reshape(1,-1)
    >>> force = vt.hammer_impulse(time, doublehit=True)
    >>> plt.plot(time.T, force.T)
    [<matplotlib.lines.Line2D object...

    """
    time_max = np.max(time)
    if imp_time is None:
        imp_time = 0.1 * time_max

    if imp_duration is None:
        imp_duration = 0.01 * time_max

    if dh_delta is None:
        dh_delta = 0.02

    dh_delta = dh_delta * time_max

    time = time.reshape(1, -1)

    imp_onset_index = int(time.shape[1] * imp_time / time_max)
    imp_offset_index = int((time.shape[1]) *
                           (imp_time + imp_duration) / time_max)
    imp_length = imp_offset_index - imp_onset_index

    T = imp_duration * 2
    omega = 2 * np.pi / T

    impulse = np.sin(omega * time[0, :imp_length])

    force = np.zeros_like(time)
    force[0, imp_onset_index:imp_onset_index + imp_length] = impulse

    if doublehit is True:
        doub_onset_index = int(time.shape[1]
                               * (imp_time + dh_delta) / time_max)
        force[0, doub_onset_index:doub_onset_index + imp_length] = impulse

    force = force / spi.simps(force.reshape(-1), dx=time[0, 1])
    return force


def decimate(t, in_signal, sample_frequency):
    r"""Decimate a signal to mimic sampling anti-aliased signal.

    Returns the signal down-sampled to `sample_frequency` with an
    anti-aliasing filter applied at 45% of ` sample_frequency`.

    Parameters
    ----------
    t : float array
        time array, size (N,)
    signal : float array
        signal array, size (N,), (m,N), or (m,N,n)
    sample_frequency : float
        new sampling frequency

    Returns
    -------
    time : float array
    decimated_signal : float array

    Examples
    --------
    >>> time = np.linspace(0,4,4096)
    >>> u = np.random.randn(1,len(time))
    >>> ttime, signal_out = decimate(time, u, 100)

    """
    dt = t[1] - t[0]
    current_frequency = 1 / dt
    freq_frac = sample_frequency / current_frequency
    Wn = .9 * freq_frac
    b, a = signal.butter(8, Wn, 'low')
    if len(in_signal.shape) > 1:
        filtered_signal = signal.lfilter(b, a, in_signal, axis=1)
    else:
        filtered_signal = signal.lfilter(b, a, in_signal)
    step = int(1 / freq_frac)
    time = t[::step]
    if len(in_signal.shape) == 1:
        filtered_signal = filtered_signal[::step]
    elif len(in_signal.shape) == 2:
        filtered_signal = filtered_signal[:, ::step]
    elif len(in_signal.shape) == 3:
        filtered_signal = filtered_signal[:, ::step, :]
    return time, filtered_signal
