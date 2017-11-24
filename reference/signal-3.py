import control as ctrl
import matplotlib.pyplot as plt
import vibrationtesting as vt
import numpy as np
sample_freq = 1e3
noise_power = 0.001 * sample_freq / 2
A = np.array([[0, 0, 1, 0],              [0, 0, 0, 1],              [-200, 100, -.2, .1],              [100, -200, .1, -.2]])
B = np.array([[0], [0], [1], [0]])
C = np.array([[35, 0, 0, 0], [0, 35, 0, 0]])
D = np.array([[0], [0]])
sys = ctrl.ss(A, B, C, D)
tin = np.arange(0, 51.2, .1)
nr=.5   # 0 is all noise on input
for i in np.arange(5): #was 2*50
    u = np.random.normal(scale=np.sqrt(noise_power), size=tin.shape)
    #print(u)
    t, yout, xout = ctrl.forced_response(sys, tin, u,rtol=1e-12)#,transpose=True)
    if 'Yout' in locals():
        Yout=np.dstack((Yout,yout+nr*np.random.normal(scale=.050*np.std(yout[0,:]), size=yout.shape)))
        Ucomb=np.dstack((Ucomb,u+(1-nr)*np.random.normal(scale=.05*np.std(u), size=u.shape)))
    else:
        Yout=yout+nr*np.random.normal(scale=.050*np.std(yout[0,:]), size=yout.shape) # 5% half the noise on output as on input
        Ucomb=u+(1-nr)*np.random.normal(scale=.05*np.std(u), size=u.shape)#(1, len(tin))) #10% noise signal on input
fig, ax = plt.subplots()
ax.plot(tin,Yout[0,:])
# [<matplotlib.lines.Line2...
Yout=Yout*np.std(Ucomb)/np.std(Yout)#40
ax.set_title('time response')
# Text(0.5,1,'time response')
freq_vec, Pxx = vt.asd(Yout, tin, windowname="hanning", ave=bool(False))
fig, ax = plt.subplots()
ax.plot(freq_vec, 20*np.log10(Pxx[0,:]))
# [<matplotlib.lines.Line2D object at ...]
ax.set_title('Raw ASDs')
# Text(0.5,1,'Raw ASDs')
freq_vec, Pxx = vt.asd(Yout, tin, windowname="hanning", ave=bool(True))
fig, ax = plt.subplots()
ax.plot(freq_vec, 20*np.log10(Pxx[0,:]))
# [<matplotlib.lines.Line2D object at ...]
ax.set_title('Averaged ASDs')
# Text(0.5,1,'Averaged ASDs')
f, Txy1, Txy2, coh, Txyv = vt.frfest(Yout, Ucomb, t,Hv=bool(True))
#fig_amp,=plt.plot(f[0,:],20*np.log10(np.abs(Txy1[0,:])),legend='$H_1$',f[0,:],20*np.log10(np.abs(Txy2[0,:])),legend='$H_2$',f[0,:],20*np.log10(np.abs(Txyv[0,:])),legend='$H_v$')
fig, ax = plt.subplots()
(line1, line2, line3) = ax.plot(f,20*np.log10(np.abs(Txy1[0,:])),f,20*np.log10(np.abs(Txy2[0,:])),f,20*np.log10(np.abs(Txyv[0,:])))
ax.set_title('FRF of ' + str(Yout.shape[2]) + ' averages.')
# Text(0.5,1,...
ax.legend((line1,line2,line3),('$H_1$','$H_2$','$H_v$'))
# <matplotlib.legend.Legend object ...>
fig, ax = plt.subplots()
ax.plot(f,180.0/np.pi*np.unwrap(np.angle(Txy1[0,:])),f,180.0/np.pi*np.unwrap(np.angle(Txy2[0,:])),f,180.0/np.pi*np.unwrap(np.angle(Txyv[0,:])))
# [<matplotlib.lines.Line2D object at ...]
ax.set_title('FRF of ' + str(Yout.shape[2]) + ' averages.')
# Text(0.5,1,...
fig, ax = plt.subplots()
ax.plot(f,coh[0,:])
# [<matplotlib.lines.Line2D object at...
vt.frfplot(f,Txy1,freq_max=3.5)
