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
# np.set_printoptions(precision=4, suppress=True)


def d2c(Ad, Bd, C, D, dt):
    """Returns continuous A, B, C, D from discrete

    Converts a set of digital state space system matrices to their
    continuous counterpart.

    Parameters
    ----------
    Ad, Bd, C, D : float arrays
        Discrete state space system matrices
    dt  : float
        Time step of discrete system

    Returns
    -------
    A, B, C, D : float arrays
        State space system matrices

    Examples
    --------
    >>> import vibrationtesting as vt
    >>> Ad = np.array([[ 0.9999,0.0001,0.01,0.],
    ...                [ 0.,0.9999,0.,0.01],
    ...                [-0.014,0.012,0.9999,0.0001],
    ...                [ 0.008, -0.014,0.0001,0.9999]])
    >>> Bd = np.array([[ 0.  ],
    ...                [ 0.  ],
    ...                [ 0.  ],
    ...                [ 0.01]])
    >>> C = np.array([[-1.4, 1.2, -0.0058, 0.0014]])
    >>> D = np.array([[-0.2]])
    >>> A, B, *_ = vt.d2c(Ad, Bd, C, D, 0.01)
    >>> print(A)
    [[-0.003   0.004   1.0001 -0.0001]
     [-0.004  -0.003  -0.      1.0001]
     [-1.4001  1.2001 -0.003   0.004 ]
     [ 0.8001 -1.4001  0.006  -0.003 ]]

    Notes
    -----
    .. note:: Zero-order hold solution
    .. note:: discrepancies between :func:`c2d` and :func:`d2c` are due to \
    typing truncation errors.
    .. seealso:: :func:`c2d`.
    """

    # Old school (very old)
    # A = la.logm(Ad) / dt
    # B = la.solve((Ad - np.eye(A.shape[0])), A) @ Bd
    sa = Ad.shape[0]
    sb = Bd.shape[1]
    AAd = np.vstack((np.hstack((Ad,                  Bd)),
                     np.hstack((np.zeros((sb, sa)),  np.eye(sb)))))
    AA = la.logm(AAd) / dt
    A = AA[0:sa, 0:sa]
    B = AA[0:sa, sa:]
    return A, B, C, D


def c2d(A, B, C, D, dt):
    """Convert continuous state system to discrete time

    Converts a set of continuous state space system matrices to their
    discrete counterpart.

    Parameters
    ----------
    A, B, C, D : float arrays
        State space system matrices
    dt : float
        Time step of discrete system

    Returns
    -------
    Ad, Bd, C, D : float arrays
        Discrete state space system matrices

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=4, suppress=True)
    >>> import vibrationtesting as vt
    >>> A1 = np.array([[ 0.,   0. ,  1.    ,  0.    ]])
    >>> A2 = np.array([[ 0.,   0. ,  0.    ,  1.    ]])
    >>> A3 = np.array([[-1.4,  1.2, -0.0058,  0.0014]])
    >>> A4 = np.array([[ 0.8, -1.4,  0.0016, -0.0038]])
    >>> A = np.array([[ 0.,   0. ,  1.    ,  0.    ],
    ...               [ 0.,   0. ,  0.    ,  1.    ],
    ...               [-1.4,  1.2, -0.0058,  0.0014],
    ...               [ 0.8, -1.4,  0.0016, -0.0038]])
    >>> B = np.array([[ 0.],
    ...               [ 0.],
    ...               [ 0.],
    ...               [ 1.]])
    >>> C = np.array([[-1.4, 1.2, -0.0058, 0.0014]])
    >>> D = np.array([[-0.2]])
    >>> Ad, Bd, *_ = vt.c2d(A, B, C, D, 0.01)
    >>> print(Ad)
    [[ 0.9999  0.0001  0.01    0.    ]
     [ 0.      0.9999  0.      0.01  ]
     [-0.014   0.012   0.9999  0.0001]
     [ 0.008  -0.014   0.0001  0.9999]]
    >>> print(Bd)
    [[ 0.  ]
     [ 0.  ]
     [ 0.  ]
     [ 0.01]]

    Notes
    -----
    .. note:: Zero-order hold solution
    .. seealso:: :func:`d2c`.
    """

    Ad, Bd, _, _, _ = sig.cont2discrete((A, B, C, D), dt)
    # Ad = la.expm(A * dt)
    # Bd = la.solve(A, (Ad - np.eye(A.shape[0]))) @ B
    return Ad, Bd, C, D


def ssfrf(A, B, C, D, omega_low, omega_high, in_index, out_index,
          num_freqs = 1000):
    """FRF of state space system

    Obtains the computed FRF of a state space system between selected input
    and output over frequency range of interest.

    Parameters
    ----------
    A, B, C, D : float arrays
                 state system matrices
    omega_low, omega_high : floats
                 low and high frequencies for evaluation
    in_index, out_index : ints
                input and output numbers (starting at 1)
    num_freqs : int
                number of frequencies at which to return FRF

    Returns
    -------
    omega : float array
            frequency vector
    H : float array
        frequency response function

    Examples
    --------
    >>> import vibrationtesting as vt
    >>> A = np.array([[ 0.,   0. ,  1.    ,  0.    ],
    ...               [ 0.,   0. ,  0.    ,  1.    ],
    ...               [-1.4,  1.2, -0.0058,  0.0014],
    ...               [ 0.8, -1.4,  0.0016, -0.0038]])
    >>> B = np.array([[ 0.],
    ...               [ 0.],
    ...               [ 0.],
    ...               [ 1.]])
    >>> C = np.array([[-1.4, 1.2, -0.0058, 0.0014]])
    >>> D = np.array([[-0.2]])
    >>> omega, H = vt.ssfrf(A, B, C, D, 0, 3.5, 1, 1)
    >>> vt.frfplot(omega, H) # doctest: +SKIP

    """
    # A, B, C, D = ctrl.ssdata(sys)
    if 0 < in_index < (B.shape[1]+1) and 0 < out_index < (C.shape[0] + 1):
        sa = A.shape[0]
        omega = np.linspace(omega_low, omega_high, num_freqs)
        H = omega * 1j
        i = 0
        for i, w in enumerate(omega):
            H[i] = (C@la.solve(w * 1j * np.eye(sa) - A, B) + D)[out_index - 1,
                                                                in_index - 1]
    else:
        raise ValueError('Input {} or output {} infeasible.'.format(in_index, out_index))
    return omega.reshape(1, -1), H.reshape(1, -1)


def sos_frf(M, C, K, Bt, Cd, Cv, Ca, omega_low, omega_high,
            in_index, out_index, num_freqs = 1000):
    """FRF of second order system

    Given second order linear matrix equation of the form
    :math:`M\\ddot{x} + C \\dot{x} + K x= \\tilde{B} u`
    and
    :math:`y = C_d x + C_v \\dot{x} + C_a\\ddot{x}`
    converts to state space form and returns the requested frequency response
    function

    Parameters
    ----------
    M, C, K, Bt, Cd, Cv, Cd : float arrays
        Mass, damping, stiffness, input, displacement sensor, velocimeter,
        and accelerometer matrices
    omega_low, omega_high : floats
        low and high frequencies for evaluation
    in_index, out_index : ints
        input and output numbers (starting at 1)
    num_freqs : int
        number of frequencies at which to return FRF


    Returns
    -------
    omega : float array
        frequency vector
    H : float array
        frequency response function

    Examples not working for second order system

    Need to make one for second order expansion

    Examples
    --------
    >>> import vibrationtesting as vt
    >>> A = np.array([[ 0.,   0. ,  1.    ,  0.    ],
    ...               [ 0.,   0. ,  0.    ,  1.    ],
    ...               [-1.4,  1.2, -0.0058,  0.0014],
    ...               [ 0.8, -1.4,  0.0016, -0.0038]])
    >>> B = np.array([[ 0.],
    ...               [ 0.],
    ...               [ 0.],
    ...               [ 1.]])
    >>> C = np.array([[-1.4, 1.2, -0.0058, 0.0014]])
    >>> D = np.array([[-0.2]])
    >>> omega, H = vt.ssfrf(A, B, C, D, 0, 3.5, 1, 1)
    >>> vt.frfplot(omega, H) # doctest: +SKIP
    """

    A, B, C, D = so2ss(M, C, K, Bt, Cd, Cv, Ca)

    omega, H = ssfrf(A, B, C, D, omega_low, omega_high, in_index, out_index,
                     num_freqs)

    return omega, H


def so2ss(M, C, K, Bt, Cd, Cv, Ca):
    """Convert second order system to state space

    Given second order linear matrix equation of the form
    :math:`M\\ddot{x} + C \\dot{x} + K x= \\tilde{B} u`
    and
    :math:`y = C_d x + C_v \\dot{x} + C_a\\ddot{x}`
    returns the state space form equations
    :math:`\\dot{z} = A z + B u`,
    :math:`y = C z + D u`

    Parameters
    ----------
    M, C, K, Bt, Cd, Cv, Cd : float arrays
        Mass , damping, stiffness, input, displacement sensor, velocimeter,
        and accelerometer matrices

    Returns
    -------
    A, B, C, D : float arrays
        State matrices

    Examples
    --------
    >>> import vibrationtesting as vt
    >>> M = np.array([[2, 1],
    ...               [1, 3]])
    >>> K = np.array([[2, -1],
    ...               [-1, 3]])
    >>> C = np.array([[0.01, 0.001],
    ...               [0.001, 0.01]])
    >>> Bt = np.array([[0], [1]])
    >>> Cd = Cv = np.zeros((1,2))
    >>> Ca = np.array([[1, 0]])
    >>> A, B, Css, D = vt.so2ss(M, C, K, Bt, Cd, Cv, Ca)
    >>> print('A: {}'.format(A))
    A:  [[ 0.      0.      1.      0.    ]
     [ 0.      0.      0.      1.    ]
     [-1.4     1.2    -0.0058  0.0014]
     [ 0.8    -1.4     0.0016 -0.0038]]
    >>> print('B: ', B)
    B:  [[ 0. ]
     [ 0. ]
     [-0.2]
     [ 0.4]]
    >>> print('C: ', Css)
    C:  [[-1.4     1.2    -0.0058  0.0014]]
    >>> print('D: ', D)
    D:  [[-0.2]]
    """

    A = np.vstack((np.hstack((np.zeros_like(M), np.eye(M.shape[0]))),
                   np.hstack((-la.solve(M, K), -la.solve(M, C)))))
    B = np.vstack((np.zeros_like(Bt), la.solve(M, Bt)))
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
                                                float(abs(pole)), float(d0),
                                                float(f0)))

        else:
            print('      {:.3f}        {:+.3f}      {:.3f}       \
                  {:.3f}         {:.3f}'.format(float(np.real(pole)),
                                                float(np.imag(pole)), float(
                                                    abs(pole)), float(d0),
                                                float(f0)))


def sos_modal(M, K, C=False, damp_diag=0.03, shift=1):
    r'''Eigen analysis of proportionally damped system.

    Optimally find mass normalized mode shapes and natural frequencies
    of a system modelled by :math:`M\ddot{x}+Kx=0`.

    If provided, obtain damping ratios presuming :math:`C` can be decoupled.

    Provides a warning if diagonalization of damping matrix fails worse than
    relative error of `damp_diag`.

    Parameters
    ----------
    M, K : float arrays
        Mass and stiffness matrices
    C : float array, optional
        Damping matrix
    damp_diag : float, optional
        Maximum amount of off-diagonal error allowed in assuming C can be
        diagonalized
    shift : float, optional
        Shift used in eigensolution. Should be approximately equal to the first
        non-zero eigenvalue.

    Returns
    -------
    omega : float array (1xN)
        Vector of natural frequencies (rad/sec)
    zeta : float array (1xN)
        Vector of damping ratios
    Psi : float array (NxN)
        Matrix of mass normalized mode shapes by column


    Examples
    --------
    >>> import numpy as np
    >>> import vibrationtesting as vt
    >>> M = np.array([[4, 0, 0],
    ...               [0, 4, 0],
    ...               [0, 0, 4]])
    >>> K = np.array([[8, -4, 0],
    ...               [-4, 8, -4],
    ...               [0, -4, 4]])
    >>> omega, zeta, Psi = vt.sos_modal(M, K, K/10)
    >>> print(omega)
    [ 0.445   1.247   1.8019]
    >>> print(Psi.T@K@Psi)
    [[ 0.1981  0.     -0.    ]
     [ 0.      1.555  -0.    ]
     [-0.     -0.      3.247 ]]

    Check that it works for rigid body modes.

    >>> K2 = K-np.eye(K.shape[0])@M*(Psi.T@K@Psi)[0,0]
    >>> omega, zeta, Psi = vt.sos_modal(M, K2)
    >>> print(omega)
    [ 0.      1.1649  1.7461]
    >>> print(Psi)
    [[-0.164   0.3685 -0.2955]
     [-0.2955  0.164   0.3685]
     [-0.3685 -0.2955 -0.164 ]]
    >>> print(np.diag(Psi.T@K2@Psi))
    [ 0.      1.3569  3.0489]

    How about non-proportional damping

    >>> C = K/10
    >>> C[0,0] = 2 * C[0,0]
    >>> omega, zeta, Psi = vt.sos_modal(M, K2, C)
    Damping matrix cannot be completely diagonalized.
    Off diagonal error of 22%.
    >>> print(omega)
    [ 0.      1.1649  1.7461]
    >>> print(zeta)
    [ 0.      0.1134  0.113 ]
    >>> print(Psi.T@C@Psi)
    [[ 0.0413 -0.0483  0.0388]
     [-0.0483  0.2641 -0.0871]
     [ 0.0388 -0.0871  0.3946]]

    '''

    K = K + M * shift

    lam, Psi = la.eigh(K, M)

    omega = np.sqrt(np.abs(lam - shift))  # round to zero

    norms = np.diag(1.0 / np.sqrt(np.diag(Psi.T@M@Psi)))

    Psi = Psi @ norms

    zeta = np.zeros_like(omega)

    if C is not False:
        diagonalized_C = Psi.T@C@Psi

        diagonal_C = np.diag(diagonalized_C)

        if min(omega) > 1e-5:
            zeta = diagonal_C / 2 / omega  # error if omega = 0
            max_off_diagonals = np.amax(np.abs(diagonalized_C
                                               - np.diag(diagonal_C)), axis=0)
            # error if no damping
            damp_error = np.max(max_off_diagonals / diagonal_C)
        else:
            zeta = np.zeros_like(omega)
            damp_error = np.zeros_like(omega)
            de_diag_C = diagonalized_C - np.diag(diagonal_C)
            for mode_num, omega_i in enumerate(omega):
                if omega[mode_num] > 1e-5:
                    zeta[mode_num] = diagonal_C[mode_num] / 2 / omega_i
                    damp_error = (np.max(np.abs(de_diag_C[:, mode_num]))
                                  / diagonal_C[mode_num])

        if damp_error > damp_diag:
            print('Damping matrix cannot be completely diagonalized.')
            print('Off diagonal error of {:4.0%}.'.format(damp_error))

    return omega, zeta, Psi


def serep(M, K, master):
    r'''System Equivalent Reduction reduced model

    Reduce size of second order system of equations by SEREP processs while
    returning expansion matrix

    Equation of the form:
    :math:`M \ddot{x} + K x = 0`
    is reduced to the form
    :math:`M_r \ddot{x}_m + Kr x_m = 0`
    where :math:`x = T x_m`, :math:`M_r= T^T M T`, :math:`K_r= T^T K T`

    Parameters
    ----------
    M, K : float arrays
        Mass and Stiffness matrices
    master : float array or list
        List of retained degrees of freedom

    Returns
    -------
    Mred, Kred, T : float arrays
        Reduced Mass matric, reduced stiffness matrix, Transformation matrix
    truncated_dofs : int list
        List of truncated degrees of freedom

    Examples
    --------
    >>> import vibrationtesting as vt
    >>> M = np.array([[4, 0, 0],
    ...               [0, 4, 0],
    ...               [0, 0, 4]])
    >>> K = np.array([[8, -4, 0],
    ...               [-4, 8, -4],
    ...               [0, -4, 4]])
    >>> retained = np.array([[1, 2]])
    >>> Mred, Kred, T, truncated_dofs = vt.serep(M, K, retained)

    Notes
    -----
    Reduced coordinate system forces can be obtained by
    `Fr = T.T @ F`

    Reduced damping matrix can be obtained using `Cr = T.T*@ C @ T`.

    If mode shapes are obtained for the reduced system, full system mode shapes
    are `phi = T @ phi_r`
    '''

    nm = int(max(master.shape))  # number of modes to keep;
    master = master.reshape(-1) - 1  # retained dofs

    ndof = int(M.shape[0])  # length(M);

    omega, _, Psi = sos_modal(M, K)
    Psi_tr = Psi[nm:, :nm]  # Truncated modes
    Psi_rr = Psi[:nm, :nm]  # Retained modes

    truncated_dofs = list(set(np.arange(ndof)) - set(master))

    T = np.zeros((ndof, nm))
    T[master, :nm] = np.eye(nm)
    T[truncated_dofs, :nm] = la.solve(Psi_rr.T, Psi_tr.T).T
    Mred = T.T @ M @T
    Kred = T.T @ K @T

    return Mred, Kred, T, np.array(truncated_dofs) + 1


def mode_expansion_from_model(Psi, omega, M, K, measured):
    r'''Deflection extrapolation to full FEM model coordinates, matrix method

    Provided an equation  of the form:

    :math:`\begin{pmatrix}-\begin{bmatrix}M_{mm}&M_{mu}\\ M_{um}&M_{uu}
    \end{bmatrix} \omega_i^2
    +\begin{bmatrix}K_{mm}&K_{mu}\\ K_{um}&K_{uu}\end{bmatrix}\end{pmatrix}`
    :math:`\begin{bmatrix}\Psi_{i_m}\\ \Psi_{i_u}\end{bmatrix}= 0`

    Where:

    - :math:`M` and :math:`K` are the mass and stiffness matrices, likely from
      a finite element model
    - :math:`\Psi_i` and :math:`\omega_i` represent a mode/frequency pair
    - subscripts :math:`m` and :math:`u` represent measure and unmeasured
      of the mode

    Determines the unknown portion of the mode (or operational deflection)
    shape, :math:`\Psi_{i_u}` by
    direct algebraic solution, aka

    :math:`\Psi_{i_u} = - (K_{uu}- M_{ss} \omega_i^2) ^{-1}
    (K_{um}-M_{um}\omega_i^2)\Psi_{i_m}`

    Parameters
    ----------
    Psi : float array
        mode shape, 2-D array
    omega : float
        natural (or driving) frequency
    M, K : float arrays
        Mass and Stiffness matrices
    measured : float or integer array or list
        List of measured degrees of freedom

    Returns
    -------
    Psi_full: float array
        Complete mode shape

    Examples
    --------
    >>> import vibrationtesting as vt
    >>> M = np.array([[4, 0, 0],
    ...               [0, 4, 0],
    ...               [0, 0, 4]])
    >>> K = np.array([[8, -4, 0],
    ...               [-4, 8, -4],
    ...               [0, -4, 4]])
    >>> measured = np.array([[1, 3]])
    >>> omega, zeta, Psi = vt.sos_modal(M, K)
    >>> Psi_measured = np.array([[-0.15], [-0.37]])
    >>> Psi_full = vt.mode_expansion_from_model(Psi_measured, omega[0], M, K, measured)
    >>> print(np.hstack((Psi[:,0].reshape(-1,1), Psi_full)))
    [[-0.164  -0.15  ]
     [-0.2955  0.2886]
     [-0.3685 -0.37  ]]

    Notes
    -----
    Reduced coordinate system forces can be obtained by
    `Fr = T.T @ F`

    Reduced damping matrix can be obtained using `Cr = T.T*@ C @ T`.

    If mode shapes are obtained for the reduced system, full system mode shapes
    are `phi = T @ phi_r`
    '''

    measured = (measured.reshape(-1) - 1)  # retained dofs
    num_measured = len(measured)
    ndof = int(M.shape[0])  # length(M);
    unmeasured_dofs = list(set(np.arange(ndof)) - set(measured))
    num_unmeasured = len(unmeasured_dofs)

    Muu = np.array(M[unmeasured_dofs, unmeasured_dofs]).reshape(num_unmeasured,
                                                                num_unmeasured)

    Kuu = np.array(K[unmeasured_dofs, unmeasured_dofs]).reshape(num_unmeasured,
                                                                num_unmeasured)
    Mum = np.array(M[unmeasured_dofs, measured]).reshape(num_unmeasured,
                                                         num_measured)
    Kum = np.array(K[unmeasured_dofs, measured]).reshape(num_unmeasured,
                                                         num_measured)

    Psi_unmeasured = la.solve((Kuu-Muu*omega**2), (Kum-Mum*omega**2)@Psi)
    Psi_full = np.zeros((num_measured+num_unmeasured, 1))
    Psi_full[measured] = Psi
    Psi_full[unmeasured_dofs] = Psi_unmeasured
    Psi_full = Psi_full.reshape(-1, 1)
    return Psi_full
