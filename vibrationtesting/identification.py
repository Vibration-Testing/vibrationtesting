"""
Created on Nov. 26, 2017
@author: Joseph C. Slater
"""
__license__ = "Joseph C. Slater"

__docformat__ = 'reStructuredText'

import numpy as np
import scipy.linalg as la
import vibration_toolbox as vtb


def sdof_cf(f, TF, Fmin=None, Fmax=None):
    """Curve fit to a single degree of freedom FRF.

    Only one peak may exist in the segment of the FRF passed to sdofcf. No
    zeros may exist within this segment. If so, curve fitting becomes
    unreliable.

    If Fmin and Fmax are not entered, the first and last elements of TF are
    used.

    Parameters
    ----------
    f: array
        The frequency vector in Hz. Does not have to start at 0 Hz.
    TF: array
        The complex transfer function
    Fmin: int
        The minimum frequency to be used for curve fitting in the FRF
    Fmax: int
        The maximum frequency to be used for curve fitting in the FRF

    Returns
    -------
    z: double
        The damping ratio
    nf: double
        Natural frequency (Hz)
    a: double
        The numerator of the identified transfer functions

        Plot of the FRF magnitude and phase.

    Examples
    --------
    >>> # First we need to load the sampled data which is in a .mat file
    >>> import vibration_toolbox as vtb
    >>> import scipy.io as sio
    >>> data = sio.loadmat(vtb.__path__[0] + '/data/case1.mat')
    >>> #print(data)
    >>> # Data is imported as arrays. Modify then to fit our function.
    >>> TF = data['Hf_chan_2']
    >>> f = data['Freq_domain']
    >>> # Now we are able to call the function
    >>> z, nf, a = vtb.sdof_cf(f,TF,500,1000)
    >>> nf
    212.092530551...

    Notes
    -----
    Author: Original Joseph C. Slater. Python version, Gabriel Loranger
    """

    # check fmin fmax existance
    if Fmin is None:
        inlow = 0
    else:
        inlow = Fmin

    if Fmax is None:
        inhigh = np.size(f)
    else:
        inhigh = Fmax

    if f[inlow] == 0:
        inlow = 1

    f = f[inlow:inhigh, :]
    TF = TF[inlow:inhigh, :]

    R = TF
    y = np.amax(np.abs(TF))
    cin = np.argmax(np.abs(TF))

    ll = np.size(f)

    w = f * 2 * np.pi * 1j

    w2 = w * 0
    R3 = R * 0

    for i in range(1, ll + 1):
        R3[i - 1] = np.conj(R[ll - i])
        w2[i - 1] = np.conj(w[ll - i])

    w = np.vstack((w2, w))
    R = np.vstack((R3, R))

    N = 2
    x, y = np.meshgrid(np.arange(0, N + 1), R)
    x, w2d = np.meshgrid(np.arange(0, N + 1), w)
    c = -1 * w**N * R

    aa1 = w2d[:, np.arange(0, N)] \
        ** x[:, np.arange(0, N)] \
        * y[:, np.arange(0, N)]
    aa2 = -w2d[:, np.arange(0, N + 1)] \
        ** x[:, np.arange(0, N + 1)]
    aa = np.hstack((aa1, aa2))

    aa = np.reshape(aa, [-1, 5])

    b, _, _, _ = la.lstsq(aa, c)

    rs = np.roots(np.array([1,
                            b[1],
                            b[0]]))
    omega = np.abs(rs[1])
    z = -1 * np.real(rs[1]) / np.abs(rs[1])
    nf = omega / 2 / np.pi

    XoF1 = np.hstack(([1 / (w - rs[0]), 1 / (w - rs[1])]))
    XoF2 = 1 / (w**0)
    XoF3 = 1 / w**2
    XoF = np.hstack((XoF1, XoF2, XoF3))

    # check if extra _ needed

    a, _, _, _ = la.lstsq(XoF, R)
    XoF = XoF[np.arange(ll, 2 * ll), :].dot(a)

    a = np.sqrt(-2 * np.imag(a[0]) * np.imag(rs[0]) -
                2 * np.real(a[0]) * np.real(rs[0]))
    Fmin = np.min(f)
    Fmax = np.max(f)
    phase = np.unwrap(np.angle(TF), np.pi, 0) * 180 / np.pi
    phase2 = np.unwrap(np.angle(XoF), np.pi, 0) * 180 / np.pi
    while phase2[cin] > 50:
        phase2 = phase2 - 360
    phased = phase2[cin] - phase[cin]
    phase = phase + np.round(phased / 360) * 360

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    fig.tight_layout()

    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.plot(f, 20 * np.log10(np.abs(XoF)), label="Identified FRF")
    ax1.plot(f, 20 * np.log10(np.abs(TF)), label="Experimental FRF")
    ax1.legend()

    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (deg)')
    ax2.plot(f, phase2, label="Identified FRF")
    ax2.plot(f, phase, label="Experimental FRF")
    ax2.legend()

    plt.show()

    a = a[0]**2 / (2 * np.pi * nf)**2
    return z, nf, a


def mdof_cf(f, TF, Fmin=None, Fmax=None):
    """
    Curve fit to multiple degree of freedom FRF

    If Fmin and Fmax are not entered, the first and last elements of TF are
    used.

    If the first column of TF is a collocated (input and output location are
    the same), then the mode shape returned is the mass normalized mode shape.
    This can then be used to generate an identified mass, damping, and
    stiffness matrix as shown in the following example.

    Parameters
    ----------
    f: array
        The frequency vector in Hz. Does not have to start at 0 Hz.
    TF: array
        The complex transfer function
    Fmin: int
        The minimum frequency to be used for curve fitting in the FRF
    Fmax: int
        The maximum frequency to be used for curve fitting in the FRF

    Returns
    -------
    z: double
        The damping ratio
    nf: double
        Natural frequency (Hz)
    u: array
        The mode shape

    Notes
    -----
    FRF are columns comprised of the FRFs presuming single input, multiple
    output z and nf are the damping ratio and natural frequency (Hz) u is the
    mode shape. Only one peak may exist in the segment of the FRF passed to
    sdofcf. No zeros may exist within this segment. If so, curve fitting
    becomes unreliable.

    Author: Original Joseph C. Slater. Python version, Gabriel Loranger


    Examples
    --------
    >>> # First we need to load the sampled data which is in a .mat file
    >>> import vibration_toolbox as vtb
    >>> import scipy.io as sio
    >>> data = sio.loadmat(vtb.__path__[0] + '/data/case2.mat')
    >>> #print(data)
    >>> # Data is imported as arrays. Modify then to fit our function
    >>> TF = data['Hf_chan_2']
    >>> f = data['Freq_domain']
    >>> # Now we are able to call the function
    >>> z, nf, a = vtb.mdof_cf(f,TF,500,1000)
    >>> nf
    192.59382330...
    """

    # check fmin fmax existance
    if Fmin is None:
        inlow = 0
    else:
        inlow = Fmin

    if Fmax is None:
        inhigh = np.size(f)
    else:
        inhigh = Fmax

    if f[inlow] == 0:
        inlow = 1

    f = f[inlow:inhigh, :]
    TF = TF[inlow:inhigh, :]

    R = TF.T

    U, _, _ = np.linalg.svd(R)
    T = U[:, 0]
    Hp = np.transpose(T).dot(R)
    R = np.transpose(Hp)

    ll = np.size(f)
    w = f * 2 * np.pi * 1j

    w2 = w * 0
    R3 = R * 0
    TF2 = TF * 0
    for i in range(1, ll + 1):
        R3[i - 1] = np.conj(R[ll - i])
        w2[i - 1] = np.conj(w[ll - i])
        TF2[i - 1, :] = np.conj(TF[ll - i, :])

    w = np.vstack((w2, w))
    R = np.hstack((R3, R))

    N = 2
    x, y = np.meshgrid(np.arange(0, N + 1), R)

    x, w2d = np.meshgrid(np.arange(0, N + 1), w)

    R = np.ndarray.flatten(R)
    w = np.ndarray.flatten(w)
    c = -1 * w**N * R

    aa1 = w2d[:, np.arange(0, N)] \
        ** x[:, np.arange(0, N)] \
        * y[:, np.arange(0, N)]
    aa2 = -w2d[:, np.arange(0, N + 1)] \
        ** x[:, np.arange(0, N + 1)]
    aa = np.hstack((aa1, aa2))

    b, _, _, _ = la.lstsq(aa, c)

    rs = np.roots(np.array([1,
                            b[1],
                            b[0]]))

    # irs = np.argsort(np.abs(np.imag(rs))) # necessary?

    omega = np.abs(rs[1])
    z = -1 * np.real(rs[1]) / np.abs(rs[1])
    nf = omega / 2 / np.pi

    XoF1 = 1 / ((rs[0] - w) * (rs[1] - w))

    XoF2 = 1 / (w**0)
    XoF3 = 1 / w**2

    XoF = np.vstack((XoF1, XoF2, XoF3)).T
    TF3 = np.vstack((TF2, TF))

    a, _, _, _ = la.lstsq(XoF, TF3)

    u = np.transpose(a[0, :])

    u = u / np.sqrt(np.abs(a[0, 0]))

    return z, nf, u
