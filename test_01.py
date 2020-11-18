import numpy as np
import functools
from poliastro.core.propagation import func_twobody
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import nbkode

t = [0, 24000]
u0 = np.array([7600,0,0,0,8,0])

k = 398600
def ad(t0, u_, k_):
    return 0, 0, 0

def fsimpler(t, u):
    x, y, z, vx, vy, vz = u
    r3 = (x ** 2 + y ** 2 + z ** 2) ** 1.5
    du = np.array([vx, vy, vz, -k * x / r3, -k * y / r3, -k * z / r3])
    return du

#f = functools.partial(func_twobody,k=k,ad=ad,ad_kwargs={})

print("start scipy")
sol = solve_ivp(fsimpler, t, u0, atol = 1e-13, rtol = 1e-12)
print("-----> end scipy")
T = sol.t
X = sol.y[0,:]
Y = sol.y[1,:]



y0 = u0
t0 = 0
print("start nbkode")
solver = nbkode.Euler(fsimpler, t0, y0)
tn, yn = solver.run(T)
print("-----> end nbkode")

XN = yn[:,0]
YN = yn[:,1]

fig, ax = plt.subplots(2,1)
ax[0].plot(T/3600,X,'b')
ax[0].plot(T/3600,XN,'r--')
ax[0].grid()
ax[1].plot(T/3600,Y,'b')
ax[1].plot(T/3600,YN,'r--')
ax[1].grid()
plt.show()
