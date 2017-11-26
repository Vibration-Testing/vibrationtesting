import matplotlib.pyplot as plt
import vibrationtesting as vt
import numpy as np
f=np.linspace(0,100,10000).reshape(-1,1);
w=f*2*np.pi;
k=1e5;m=1;c=1;
frf1=1./(m*(w*1j)**2+c*1j*w+k)
frf2=1./(m*(w*1j)**2+c*1j*w+k*3)
vt.frfplot(f,np.hstack((frf1,frf2)), legend = ['FRF 1','FRF 2'])
                                     # doctest: +SKIP
