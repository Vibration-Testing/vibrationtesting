import control as ctrl
import matplotlib.pyplot as plt
import vibrationtesting as vt
import numpy as np
sample_freq = 1e3
noise_power = 0.001 * sample_freq / 2
A = np.array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [-200, 100, -.2, .1],
              [100, -200, .1, -.2]])
B = np.array([[0], [0], [1], [0]])
C = np.array([[35, 0, 0, 0], [0, 35, 0, 0]])
D = np.array([[0], [0]])
sys = ctrl.ss(A, B, C, D)
tin = np.arange(0, 51.2, .1)
nr = .5   # 0 is all noise on input
for i in np.arange(520):
    u = np.random.normal(scale=np.sqrt(noise_power), size=tin.shape)
    #print(u)
    t, yout, xout = ctrl.forced_response(sys, tin, u,rtol=1e-12)
    if 'Yout' in locals():
        Yout=np.dstack((Yout,yout
                +nr*np.random.normal(scale=.050*np.std(yout[0,:]),
                 size=yout.shape)))
        Ucomb=np.dstack((Ucomb,u+(1-nr)
                *np.random.normal(scale=.05*np.std(u),
                 size=u.shape)))
    else:
        Yout=yout+nr*np.random.normal(scale=.05*np.std(yout[0,:]),
                  size=yout.shape)
                  # noise on output is 5% scale of input
        Ucomb=u+(1-nr)*np.random.normal(scale=.05*np.std(u),
                  size=u.shape)#(1, len(tin)))
                  # 5% noise signal on input
f, Hxy1, Hxy2, coh, Hxyv = vt.frfest(Yout, Ucomb, t, Hv=bool(True))
vt.frfplot(f,Hxy2,freq_max=3.5, legend=['$H_{11}$', '$H_{12}$'])
              # doctest: +SKIP
vt.frfplot(f, np.vstack((Hxy1[0,:], Hxy2[0,:], Hxyv[0,:])),
              legend=['$H_{11-1}$','$H_{11-2}$','$H_{11-v}$'])
              # doctest: +SKIP
