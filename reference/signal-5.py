import numpy as np
import vibrationtesting as vt
import matplotlib.pyplot as plt
sample_freq = 1e3
tfinal = 5
fs = 100
A = 10
freq = 5
noise_power = 0.001 * sample_freq / 2
time = np.reshape(np.arange(0, tfinal, 1/sample_freq),(1,-1))
xsin = A*np.sin(2*np.pi*freq*time)
xcos = A*np.cos(2*np.pi*freq*time)
x=np.dstack((xsin,xcos)) # assembling individual records. vstack
xw=vt.hanning(x)*x
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(time.T,x[:,:,1].T)
# [<matplotlib.lines.Line2D object at ...>]
ax1.set_ylim([-20, 20])
# (-20, 20)
ax1.set_title('Unwindowed data, 2 records.')
# Text(0.5,1,'Unwindowed data, 2 records.')
ax1.set_ylabel('$x(t)$')
# Text(0,0.5,'$x(t)$')
ax2.plot(time[0,:],xw[0,:],time[0,:],vt.hanning(x)[0,:]*A,
                     '--',time[0,:],-vt.hanning(x)[0,:]*A,'--')
# [<matplotlib.lines.Line2D object at ...>]
ax2.set_ylabel('Hanning windowed $x(t)$')
# Text(0,0.5,'Hanning windowed $x(t)$')
ax2.set_xlabel('time')
# Text(0.5,0,'time')
ax2.set_title('Effect of window. Note the scaling to conserve ASD amplitude')
# Text(0.5,1,'Effect of window. Note the scaling to conserve ASD amplitude')
fig.tight_layout()
