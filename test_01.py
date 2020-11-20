import numpy as np
import functools
from poliastro.core.propagation import func_twobody
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from nbkode import RungeKutta45, DOP853


tend = 100                     # Finish time [s]

t = [0, tend]                   # Time begin and end for scipy
u0 = np.array([7600,0,0,0,8,0]) # Initial condition vector (for both)

k = 398600                      # grav. param

def ad(t0, u_, k_):
    return 0, 0, 0
f = functools.partial(func_twobody,k=k,ad=ad,ad_kwargs={})

def fsimpler(t, u):
    x, y, z, vx, vy, vz = u
    r3 = (x ** 2 + y ** 2 + z ** 2) ** 1.5
    du = np.array([vx, vy, vz, -k * x / r3, -k * y / r3, -k * z / r3])
    return du

print("start scipy")
sol = solve_ivp(f, t, u0, atol = 1e-12, rtol = 1e-11 ,method = "DOP853")
print("-----> end scipy")

print("start nbkode")
solver = RungeKutta45(func_twobody, 0, u0, atol = 1e-12, rtol = 1e-11, params = (k,ad,{}))
tn = np.arange(0,tend,10)
tn, yn = solver.run(tn)
print("-----> end nbkode")


# scipy solution
T = sol.t
X = sol.y[0,:]
Y = sol.y[1,:]

# nbkode solution
TN = tn
XN = yn[:,0]
YN = yn[:,1]

fct = 1

fig, ax = plt.subplots(2,1)
ax[0].plot(T/fct,X,'b')
ax[0].plot(TN/fct,XN,'r--')
ax[0].grid()
ax[1].plot(T/fct,Y,'b')
ax[1].plot(TN/fct,YN,'r--')
ax[1].grid()

plt.figure()
plt.plot(X,Y,'b')
plt.plot(XN,YN,'r--')
plt.show()
