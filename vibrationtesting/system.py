"""
Created on 14 May 2015
@author: Joseph C. Slater
"""
__license__ = "Joseph C. Slater"

__docformat__ = 'reStructuredText'


import math
import warnings

import numpy as np
# import control as ctrl
import scipy.signal as sig
import scipy.linalg as la


def d2c(Ad, Bd, C, D, dt):
    """Returns continuous A, B, C, D from discrete A, B, C, D
    Converts a set of digital state space system matrices to their
    continuous counterpart.
    """
    A = la.logm(Ad) / dt
    B = la.solve((Ad - np.eye(A.shape[0])), A) @ Bd
    return A, B, C, D


def c2d(A, B, C, D, dt):
    """Returns discrete Ad, Bd, C, D from continuous A, B, C, D
    Converts a set of digital state space system matrices to their
    continuous counterpart.
    Simply calls scipy.signal.cont2discrete
    """

    Ad, Bd, _, _, _ = sig.cont2discrete((A, B, C, D), dt)
    Ad = la.expm(A * dt)
    Bd = la.solve(A, (Ad - np.eye(A.shape[0]))) @ B
    return Ad, Bd, C, D


def ssfrf(A, B, C, D, omega_low, omega_high, in_index, out_index):
    """returns omega, H
    obtains the computed FRF of a state space system between selected input
    and output over frequency range of interest.
    """
    # A, B, C, D = ctrl.ssdata(sys)
    sa = A.shape[0]
    omega = np.linspace(omega_low, omega_high, 1000)
    H = omega * 1j
    i = 0
    for i in np.arange(len(omega)):
        w = omega[i]
        H[i] = (C@la.solve(w * 1j * np.eye(sa) - A, B) + D)[out_index,
                                                            in_index]
    return omega, H


def so2ss(M, C, K, Bt, Cd, Cv, Ca):
    """returns A, B, C, D
    Given second order linear matrix equation of the form
    :math:`M\\ddot{x} + C \\dot{x} + K x= \\tilde{B} u`
    and
    :math:`y = C_d x + + C_v \\dot{x} + C_a\\ddot{x}`
    returns the state space form equations
    :math:`\\dot{z} = A z + B u`,
    :math:`y = C z + D u`

    Parameters
    ----------
    M: array
        Mass matrix
    C: array
        Damping matrix
    K:  array
        Stiffness matrix
    Bt: array
        Input matrix
    Cd: array
        Displacement sensor output matrix
    Cv: array
        Velocimeter output matrix
    Ca: array
        Accelerometer output matrix

    Returns
    -------
    A: array
        State matrix
    B: array
        Input matrix
    C: array
        Output matrix
    D: array
        Pass through matrix

    Examples
    --------
    >>> import numpy as np
    >>> import vibrationtesting as vt
    >>> M = np.array([[2, 1],[1, 3]])
    >>> K = np.array([[2, -1],[-1, 3]])
    >>> C = np.array([[0.01, 0.001],[0.001, 0.01]])
    >>> Bt = np.array([[0],[1]])
    >>> Cd = Cv = np.zeros((1,2))
    >>> Ca = np.array([[1, 0]])
    >>> A, B, Css, D = vt.so2ss(M, C, K, Bt, Cd, Cv, Ca)
    >>> np.set_printoptions(precision=4, suppress = True)
    >>> print('A: {}'.format(A))
    A:  [[ 0.      0.      1.      0.    ]
     [ 0.      0.      0.      1.    ]
     [-1.4     1.2    -0.0058  0.0014]
     [ 0.8    -1.4     0.0016 -0.0038]]
    >>> print('B: ', B)
    B:  [[ 0.]
     [ 0.]
     [ 0.]
     [ 1.]]
    >>> print('C: ', Css)
    C:  [[-1.4     1.2    -0.0058  0.0014]]
    >>> print('D: ', D)
    D:  [[-0.2]]
    """

    A = np.vstack((np.hstack((np.eye(2) * 0, np.eye(2))),
                   np.hstack((-la.solve(M, K), -la.solve(M, C)))))
    B = np.vstack((np.zeros((Bt.shape[0], 1)), Bt))
    C_ss = np.hstack((Cd - Ca@la.solve(M, K), Cv - Ca@la.solve(M, C)))
    D = Ca@la.solve(M, Bt)

    return A, B, C_ss, D


def damp(A):
    '''Display natural frequencies and damping ratios of state matrix.
    '''

    # Original Author: Kai P. Mueller <mueller@ifr.ing.tu-bs.de> for Octave
    # Created: September 29, 1997.

    print("............... Eigenvalue ...........     Damping     Frequency")
    print("--------[re]---------[im]--------[abs]----------------------[Hz]")
    e, _ = la.eig(A)

    for i in range(len(e)):
        pole = e[i]

        d0 = -np.cos(math.atan2(np.imag(pole), np.real(pole)))
        f0 = 0.5 / np.pi * abs(pole)
        if (abs(np.imag(pole)) < abs(np.real(pole))):
            print('      {:.3f}                    {:.3f}       \
                  {:.3f}         {:.3f}'.format(float(np.real(pole)),
                  float(abs(pole)), float(d0), float(f0)))

        else:
            print('      {:.3f}        {:+.3f}      {:.3f}       \
                  {:.3f}         {:.3f}'.format(float(np.real(pole)),
                  float(np.imag(pole)), float(abs(pole)), float(d0),
                  float(f0)))
