from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import vibrationtesting as vt
import numpy.linalg as la

# Generate a 5 second test signal, a 10 V sine wave at 50 Hz, corrupted by
# 0.001 V**2/Hz of white noise sampled at 1 kHz.

sample_freq = 1e3
tfinal = 5
sig_freq=50
A=10
noise_power = 0.0001 * sample_freq / 2
noise_power = A/1e12
time = np.arange(0,tfinal,1/sample_freq)
time = np.reshape(time, (1, -1))
x = A*np.sin(2*np.pi*sig_freq*time)
x = x + np.random.normal(scale=np.sqrt(noise_power), size=(1, time.shape[1]))
fig, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(time[0,:],x[0,:])
# [<matplotlib.lines.Line2D object at ...>]
ax1.set_title('Time history')
# Text(0.5,1,'Time history')
ax1.set_xlabel('Time (sec)')
# Text(0.5,0,'Time (sec)')
ax1.set_ylabel('$x(t)$')
# Text(0,0.5,'$x(t)$')

# Compute and plot the autospectrum density.

freq_vec, Pxx = vt.asd(x, time, windowname="hanning", ave=bool(False))
ax2.plot(freq_vec, 20*np.log10(Pxx[0,:]))
# [<matplotlib.lines.Line2D object at ...>]
ax2.set_ylim([-400, 100])
# (-400, 100)
ax2.set_xlabel('frequency (Hz)')
# Text(0.5,0,'frequency (Hz)')
ax2.set_ylabel('PSD (V**2/Hz)')
# Text(0,0.5,'PSD (V**2/Hz)')

# If we average the last half of the spectral density, to exclude the
# peak, we can recover the noise power on the signal.
