import matplotlib.pyplot as plt
import vibrationtesting as vt
import numpy as np
f=np.linspace(0,100,10000).reshape(-1,1);
w=f*2*np.pi;
k=1e5;m=1;c=1;
tf=1./(m*(w*1j)**2+c*1j*w+k)
vt.frfplot(f,tf)
vt.frfplot(f,tf,5)

# Copyright J. Slater, Dec 17, 1994
# Updated April 27, 1995
# Ported to Python, July 1, 2015
