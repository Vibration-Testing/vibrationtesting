"""
System manipulation, reduction, and corrections functions.

@author: Joseph C. Slater
"""
__license__ = "Joseph C. Slater"

__docformat__ = 'reStructuredText'


import math

import numpy as np
import scipy.signal as sig
import scipy.linalg as la


def d2c(Ad, Bd, C, D, dt):
    r"""Return continuous A, B, C, D from discrete form.

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
    AAd = np.vstack((np.hstack((Ad, Bd)),
                     np.hstack((np.zeros((sb, sa)), np.eye(sb)))))
    AA = la.logm(AAd) / dt
    A = AA[0:sa, 0:sa]
    B = AA[0:sa, sa:]
    return A, B, C, D


def c2d(A, B, C, D, dt):
    """Convert continuous state system to discrete time.

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
          num_freqs=1000):
    """Return FRF of state space system.

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
    if 0 < in_index < (B.shape[1] + 1) and 0 < out_index < (C.shape[0] + 1):
        sa = A.shape[0]
        omega = np.linspace(omega_low, omega_high, num_freqs)
        H = omega * 1j
        i = 0
        for i, w in enumerate(omega):
            H[i] = (C@la.solve(w * 1j * np.eye(sa) - A, B) + D)[out_index - 1,
                                                                in_index - 1]
    else:
        raise ValueError(
            'Input {} or output {} infeasible.'.format(in_index, out_index))
    return omega.reshape(1, -1), H.reshape(1, -1)


def sos_frf(M, C, K, Bt, Cd, Cv, Ca, omega_low, omega_high,
            in_index, out_index, num_freqs=1000):
    r"""Return FRF of second order system.

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
    r"""Convert second order system to state space.

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
    """Display natural frequencies and damping ratios of state matrix."""
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
    r"""Eigen analysis of proportionally damped system.

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

    """
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
    r"""System Equivalent Reduction Expansion Process reduced model.

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
        List of truncated degrees of freedom, zero indexed

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

    .. seealso:: :func:`guyan`

    """
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

    return Mred, Kred, T, np.array(truncated_dofs)


def guyan(M, K, master=None, fraction=None):
    r"""Guyan reduced model.

    Applies Guyan Reductions to second order system of equations of the form

    .. math:: M \ddot{x} + K x = 0

    which are reduced to the form

    .. math:: M_r \ddot{x}_m + K_r x_m = 0

    where

    .. math::

        x = T x_m

        M_r= T^T M T

        K_r= T^T K T

    Parameters
    ----------
    M, K : float arrays
        Mass and Stiffness matrices
    master : float array or list, optional
        List of retained degrees of freedom (0 indexing)
    fraction : float, optional
        Fraction of degrees of freedom (0< `fraction` <1.0) to retain in model.
        If both master and fraction are neglected, fraction is set to 0.25.

    Returns
    -------
    Mred, Kred, T : float arrays
        Reduced Mass matric, Reduced Stiffness matrix, Transformation matrix
    master_dofs : int list
        List of master degrees of freedom (0 indexing)
    truncated_dofs : int list
        List of truncated degrees of freedom (0 indexing)

    Examples
    --------
    >>> import vibrationtesting as vt
    >>> M = np.array([[4, 0, 0],
    ...               [0, 4, 0],
    ...               [0, 0, 4]])
    >>> K = np.array([[8, -4, 0],
    ...               [-4, 8, -4],
    ...               [0, -4, 4]])
    >>> Mred, Kred, T, master, truncated_dofs = vt.guyan(M, K, fraction = 0.5)


    Notes
    -----
    Reduced coordinate system forces can be obtained by `Fr = T.T @ F`.

    Reduced damping matrix can be obtained using `Cr = T.T @ C @ T`.

    If mode shapes are obtained for the reduced system, full system mode shapes
    are `phi = T @ phi_r`.

    Code is not as efficient as possible. Using block submatrices would be
    more efficient.

    """
    if master is None:
        if fraction is None:
            fraction = 0.25

        ratios = np.diag(M) / np.diag(K)
        ranked = [i[0] for i in sorted(enumerate(ratios), key=lambda x: x[1])]
        thresh = int(fraction * ratios.size)
        if (thresh >= ratios.size) or thresh == 0:
            print("Can't keep", thresh, 'DOFs.')
            print("Fraction of", fraction, "is too low or too high.")
            return 0, 0, 0, 0, 0

        master = ranked[-thresh:]

    master = np.array(master)
    nm = master.size  # number of dofs to keep;
    master = master.reshape(-1)   # retained dofs

    ndof = int(M.shape[0])  # length(M);

    truncated_dofs = list(set(np.arange(ndof)) - set(master))
    # truncated_dofs = np.array(set(np.arange(ndof)) - set(master))
    """
    Mmm = M[master].T[master].T
    Kmm = K[master].T[master].T
    Mtm = M[truncated_dofs].T[master].T
    Ktm = K[truncated_dofs].T[master].T
    Mtt = M[truncated_dofs].T[truncated_dofs].T
    Ktt = K[truncated_dofs].T[truncated_dofs].T"""

    # Mmm = slice(M, master, master)
    # Kmm = slice(K, master, master)
    # Mtm = slice(M, truncated_dofs, master)
    Ktm = slice(K, truncated_dofs, master)
    # Mtt = slice(M, truncated_dofs, truncated_dofs)
    Ktt = slice(K, truncated_dofs, truncated_dofs)

    T = np.zeros((ndof, nm))
    T[master, :nm] = np.eye(nm)
    T[truncated_dofs, :nm] = la.solve(-Ktt, Ktm)
    Mred = T.T @ M @ T
    Kred = T.T @ K @ T
    return Mred, Kred, T, master, truncated_dofs


def mode_expansion_from_model(Psi, omega, M, K, measured):
    r"""Deflection extrapolation to full FEM model coordinates, matrix method.

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
        mode shape or shapes, 2-D array columns of which are mode shapes
    omega : float or 1-D float array
        natural (or driving) frequencies
    M, K : float arrays
        Mass and Stiffness matrices
    measured : float or integer array or list
        List of measured degrees of freedom (0 indexed)

    Returns
    -------
    Psi_full : float array
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
    >>> measured = np.array([[0, 2]])
    >>> omega, zeta, Psi = vt.sos_modal(M, K)
    >>> Psi_measured = np.array([[-0.15], [-0.37]])
    >>> Psi_full = vt.mode_expansion_from_model(Psi_measured, omega[0], M, K,
    ... measured)
    >>> print(np.hstack((Psi[:,0].reshape(-1,1), Psi_full)))
    [[-0.164  -0.15  ]
     [-0.2955  0.2886]
     [-0.3685 -0.37  ]]

    Notes
    -----
    .. seealso:: incomplete multi-mode update. Would require each at a
      different frequency.

    """
    measured = measured.reshape(-1)  # retained dofs
    num_measured = len(measured)
    ndof = int(M.shape[0])  # length(M);
    unmeasured_dofs = list(set(np.arange(ndof)) - set(measured))
    num_unmeasured = len(unmeasured_dofs)

    # Code from before my slicing code
    """
    Muu = np.array(M[unmeasured_dofs].T[unmeasured_dofs].T).
                   reshape(num_unmeasured, num_unmeasured)

    Kuu = np.array(K[unmeasured_dofs].T[unmeasured_dofs].T)
                   .reshape(num_unmeasured, num_unmeasured)
    Mum = np.array(M[unmeasured_dofs].T[measured].T).reshape(num_unmeasured,
                                                         num_measured)
    Kum = np.array(K[unmeasured_dofs].T[measured].T).reshape(num_unmeasured,
                                                         num_measured)
    """

    Muu = slice(M, unmeasured_dofs, unmeasured_dofs)
    Kuu = slice(K, unmeasured_dofs, unmeasured_dofs)
    Mum = slice(M, unmeasured_dofs, measured)
    Kum = slice(K, unmeasured_dofs, measured)

    if isinstance(omega, float):
        omega = np.array(omega).reshape(1)

    Psi_full = np.zeros((num_measured + num_unmeasured, Psi.shape[1]))
    Psi_full[measured] = Psi

    for i, omega_n in enumerate(omega):
        Psi_i = Psi[:, i].reshape(-1, 1)
        Psi_unmeasured = la.solve((Kuu - Muu * omega_n**2),
                                  (Kum - Mum * omega_n**2)@Psi_i)
        Psi_full[unmeasured_dofs, i] = Psi_unmeasured
        # Psi_full = Psi_full.reshape(-1, 1)
    return Psi_full


def improved_reduction(M, K, master=None, fraction=None):
    """Incomplete.

    4.14 Friswell

    """
    print('not written yet')
    return


def model_correction_direct(Psi, omega, M, K, method='Baruch'):
    """Direct model updating using model data.

    Parameters
    ----------
    Psi : float array
        Expanded mode shapes from experimental measurements. Must be
        real-valued. See `mode_expansion_from_model`.
    omega : float array
        Natural frequencies identified from modal analysis (diagonal matrix or
        vector).
    M, K : float arrays
        Analytical mass and stiffness matrices
    method : string, optional
        `Baruch` and Bar-Itzhack [1]_ or `Berman` and Nagy [2]_
        (default Baruch)

    Returns
    -------
    Mc, Kc : float arrays
        Corrected mass and stiffness matrices.

    Notes
    -----
    .. [1] Baruch, M. and Bar-Itzhack, I.Y., "Optimal Weighted
       Orthogonalization of Measured Modes," *AIAA Journal*, 16(4), 1978, pp.
       346-351.
    .. [2] Berman, A. and Nagy, E.J., 1983, "Improvements of a Large Analytical
       Model using Test Data," *AIAA Journal*, 21(8), 1983, pp. 1168-1173.

    """
    if len(omega.shape) == 1:
        omega = np.diag(omega)
    elif omega.shape[0] != omega.shape[1]:
        omega = np.diag(omega)

    lam = omega @ omega

    if method is 'Berman':

        Mdiag = Psi.T@M@Psi
        eye_size = Mdiag.shape[0]
        Mc = (M + M @ Psi @ la.solve(Mdiag, np.eye(eye_size) - Mdiag)
              @ la.solve(Mdiag, Psi.T) @ M)

        Kc = (K - K @ Psi @ Psi.T @ M
              - M @ Psi @ Psi.T @ K
              + M @Psi@ Psi.T @ K @ Psi @ Psi.T @ M
              + M @ Psi @ lam @ Psi.T @ M)

    else:  # Defaults to Baruch method.
        Phi = rsolve(la.sqrtm(Psi.T @ M @ Psi), Psi, assume_a='pos')

        PhiPhiT = Phi@Phi.T

        sec_term = K @ PhiPhiT @ M

        Kc = K - sec_term - sec_term.T \
            + M @ PhiPhiT @ K @ PhiPhiT @ M\
            + M @ Phi @ lam @ Phi.T @ M

        Mc = M

    return Mc, Kc


def slice(Matrix, a, b):
    """Slice a matrix properly- like Octave.

    Addresses the confounding inconsistency that `M[a,b]` acts differently if
    `a` and `b` are the same length or different lengths.

    Parameters
    ----------
    Matrix : float array
        Arbitrary array
    a, b : int lists or arrays
        list of rows and columns to be selected from `Matrix`

    Returns
    -------
    Matrix : float array
        Properly sliced matrix- no casting allowed.

    """
    # a = a.reshape(-1)
    # b = b.reshape(-1)

    return (Matrix[np.array(a).reshape(-1, 1), b]
            .reshape(np.array(a).shape[0], np.array(b).shape[0]))


def rsolve(B, C, **kwargs):
    """Solve right Gauss elimination equation.

    Given :math:`A B  = C` return :math:`A = C B^{-1}`

    This uses `scipy.linalg.solve` with a little matrix manipulation first.
    All keyword arguments of `scipy.linalg.solve` may be used.

    Parameters
    ----------
    B, C : float arrays

    Returns
    -------
    A : float array

    Examples
    --------
    >>> import numpy as np
    >>> import vibrationtesting as vt
    >>> B = np.array([[ 8, -4,  0],
    ...               [-4,  8, -4],
    ...               [ 0, -4,  4]])
    >>> C = np.array([[ 32, -16,   0],
    ...            [-16,  36, -20],
    ...            [  4, -24,  20]])
    >>> A = vt.rsolve(B, C)
    >>> print(np.round(rsolve(B, C)))
    [[ 4.  0.  0.]
     [-0.  4. -1.]
     [ 0. -1.  4.]]

    Notes
    -----
    .. seealso:: `scipy.linalg.solve`

    """
    return la.solve(B.T, C.T, **kwargs).T


def real_modes(Psi, autorotate = True):
    r"""Real modes from complex modes.

    Assuming a transformation

    .. math:: Psi_{real} = Psi_{complex} T

    exists, where :math:`T` is a complex transformation matrix, find
    :math:`Psi_{real}` using linear algebra.

    Parameters
    ----------
    Psi : complex float array
        Complex mode shapes (displacement)
    autorotate : Boolean, optional
        Attempt to rotate to near-real first

    Returns
    -------
    Psi : float array
        Real modes

    Examples
    --------
    >>> import vibrationtesting as vt
    >>> M = np.array([[4, 0, 0],
    ...               [0, 4, 0],
    ...               [0, 0, 4]])
    >>> Cso = np.array([[.1,0,0],
    ...                 [0,0,0],
    ...                 [0,0,0]])
    >>> K = np.array([[8, -4, 0],
    ...               [-4, 8, -4],
    ...               [0, -4, 4]])
    >>> Bt = np.array([[1],[0],[0]])
    >>> Ca = np.array([[1,0,0]])
    >>> Cd = Cv = np.zeros_like(Ca)
    >>> A, B, C, D = vt.so2ss(M, Cso, K, Bt, Cd, Cv, Ca)
    >>> Am, Bm, Cm, Dm, eigenvalues, modes = vt.ss_modal(A, B, C, D)
    >>> Psi = vt.real_modes(modes[:,0::2])

    Notes
    -----
    .. note:: Rotation of modes should be performed to get them as close to real
      as possible first.
    .. warning:: Current autorotate bases the rotation on de-rotating the first
      element of each vector. User can use their own pre-process by doing to
      and setting `autorotate` to False.

    """
    if autorotate is True:
        Psi = Psi@np.diag(np.exp(np.angle(Psi[0, :]) * -1j))
    Psi_real = np.real(Psi)
    Psi_im = np.imag(Psi)

    Psi = Psi_real - Psi_im @ la.lstsq(Psi_real, Psi_im)[0]
    return Psi


def ss_modal(A, B = None, C = None, D = None):
    r"""State space modes, frequencies, damping ratios, and modal matrices.

    Parameters
    ----------
    A, B, C, D : float arrays
        State space system matrices

    Returns
    -------
    Am, Bm, Cm, Dm : float arrays
        Modal state space system matrices

    Examples
    --------
    >>> import vibrationtesting as vt
    >>> M = np.array([[4, 0, 0],
    ...               [0, 4, 0],
    ...               [0, 0, 4]])
    >>> Cso = np.array([[.1,0,0],
    ...                 [0,0,0],
    ...                 [0,0,0]])
    >>> K = np.array([[8, -4, 0],
    ...               [-4, 8, -4],
    ...               [0, -4, 4]])
    >>> Bt = np.array([[1],[0],[0]])
    >>> Ca = np.array([[1,0,0]])
    >>> Cd = Cv = np.zeros_like(Ca)
    >>> A, B, C, D = vt.so2ss(M, Cso, K, Bt, Cd, Cv, Ca)
    >>> Am, Bm, Cm, Dm, eigenvalues, modes = vt.ss_modal(A, B, C, D)
    >>> np.allclose(Am,np.array(
    ... [[-0.0013+0.445j,  0.0000+0.j,      0.0000+0.j,      0.0000+0.j,      0.0000+0.j,
    ...      0.0000+0.j    ],
    ... [ 0.0000+0.j,     -0.0013-0.445j,   0.0000+0.j,      0.0000+0.j,      0.0000+0.j,
    ...   0.0000+0.j    ],
    ... [ 0.0000+0.j,      0.0000+0.j,     -0.0068+1.247j,   0.0000+0.j,      0.0000+0.j,
    ...   0.0000+0.j    ],
    ... [ 0.0000+0.j,      0.0000+0.j,      0.0000+0.j,     -0.0068-1.247j,
    ...   0.0000+0.j,      0.0000+0.j    ],
    ... [ 0.0000+0.j,      0.0000+0.j,      0.0000+0.j,      0.0000+0.j,
    ...  -0.0044+1.8019j,  0.0000+0.j    ],
    ... [ 0.0000+0.j,      0.0000+0.j,      0.0000+0.j,      0.0000+0.j,      0.0000+0.j,
    ...  -0.0044-1.8019j]]), atol=0.001)
    True
    >>> Cm  # doctest: +SKIP
     [[ 0.0594-0.0001j 0.0594+0.0001j 0.0039-0.717j 0.0039+0.717j
        0.0241-0.9307j  0.0241+0.9307j]]


    """
    if B is None:
        B = np.zeros_like(A)

    if C is None:
        C = np.zeros_like(A)

    if D is None:
        D = np.zeros_like(A)

    eigenvalues, vectors = la.eig(A)

    idxp = abs(eigenvalues).argsort()
    eigenvalues = eigenvalues[idxp]
    vectors = vectors[:, idxp]

    A_modal = la.solve(vectors,A)@vectors
    B_modal = la.solve(vectors,B)
    C_modal = C@vectors
    # D_modal = D wasted CPUs
    return A_modal, B_modal, C_modal, D, eigenvalues, vectors
