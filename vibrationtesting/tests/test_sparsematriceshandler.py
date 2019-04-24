"""pytest unit tests for vibrationtesting"""

import numpy as np
import vibrationtesting as vt
import numpy.testing as nt
import scipy.io as sio

def test_sos_modal_forsparse():
	mat_contents=sio.loadmat('vibrationtesting/data/WingBeamforMAC.mat') # WFEM generated .mat file
	K = (mat_contents['K'])
	M = (mat_contents['M'])
	## Mr and Kr are WFEM outputs after Guyan reduction
	Kr = (mat_contents['Kr'])
	Mr = (mat_contents['Mr'])
	master = np.array([[ 2,  3,  4,  5,  6,  8,  9, 10, 11, 12, 14, 15, 16, 17, 18, 20,
		21, 22, 23, 24, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38, 39,
		40, 41, 42, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 56, 57, 58,
		59, 60, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 74, 75, 76, 77,
		78, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90]])
	## Mred and Kred are from guyan_forsparse
	[Mred, Kred, T, master] = vt.guyan_forsparse(M, K, master=master, fraction=None)
	[omega_sp,zeta_sp,Psi_sp] = vt.sos_modal_forsparse(Mred,Kred)
	Kbm = Kr.todense()
	Mbm = Mr.todense()
	omega, zeta, Psi = vt.sos_modal(Mbm, Kbm)
	## The below compares sparsematriceshandler.py vs system.py results
	nt.assert_array_almost_equal(omega_sp,omega)
	nt.assert_array_almost_equal(zeta_sp,zeta)
	nt.assert_array_almost_equal(Psi_sp,Psi)

def test_guyan_forsparse():
	mat_contents=sio.loadmat('vibrationtesting/data/WingBeamforMAC.mat') # WFEM generated .mat file
	K = (mat_contents['K'])
	M = (mat_contents['M'])
	## Mr and Kr are WFEM outputs after Guyan reduction
	Kr = (mat_contents['Kr'])
	Mr = (mat_contents['Mr'])
	master = np.array([[ 2,  3,  4,  5,  6,  8,  9, 10, 11, 12, 14, 15, 16, 17, 18, 20,
		21, 22, 23, 24, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38, 39,
		40, 41, 42, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54, 56, 57, 58,
		59, 60, 62, 63, 64, 65, 66, 68, 69, 70, 71, 72, 74, 75, 76, 77,
		78, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90]])
	## Mred and Kred are from guyan_forsparse
	[Mred, Kred, T, master] = vt.guyan_forsparse(M, K, master=master, fraction=None)
	Kred = Kred.todense()
	Mred = Mred.todense()
	Kr = Kr.todense()
	Mr = Mr.todense()
	## The below compares the two outputs guyanWFEM vs guyanVibrationtesting
	nt.assert_array_almost_equal(Mred,Mr)
	nt.assert_array_almost_equal(Kred,Kr)
