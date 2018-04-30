__author__ = 'julie'
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


"""
Minimize f(x) = 0.5 x^T A x - b^T x
using steepest descent method
"""
mu = 1.0
L  = 4.0
A = np.diag(np.array([mu,L]))
b = np.array([0.,0.])

# function, derivative, analytical steplength
f = lambda x: 0.5*x.dot(A.dot(x)) - b.dot(x)
df= lambda x: A.dot(x) - b
steplength = lambda p: p.dot(p)/p.dot(A.dot(p))

# random starting point
phi_rand = 2*np.pi*np.random.rand()
x0 = 3.0*np.array([np.sin(phi_rand),np.cos(phi_rand)])
# worst starting point
#x0 = np.array([1./mu, 1./L])
x0 = 3.0*x0/np.linalg.norm(x0)
#x0 = 1./np.diag(A)

# analytical solution: we know it in this case
x_opt = np.linalg.solve(A,b)

# tolerance: stop when ||x-x_opt|| < tol
# in practice of course one stope when, e.g., |df(x)|<tol or similar
tol = 1.0E-06

x = x0
i = 0

x_history = np.zeros((2,0))
f_history = np.zeros((1,0))
x_history = np.append(x_history,np.reshape(x,(2,1)),axis=1)
f_history = np.append(f_history,f(x))

while np.linalg.norm(x-x_opt) > tol:
    # steepest descent direction
    p = -df(x)
    print("Iter: %3d, f=%15.6e, |grad f|=%15.6e, |x-x_opt|=%15.6e" % \
          (i, f(x), np.linalg.norm(p,2), np.linalg.norm(x-x_opt,2)))

    if np.linalg.norm(p) == 0.0:
        # success
        break
    # steplength: in this case we can determin it analytically
    alpha = steplength(p)
    x = x + alpha*p
    i = i+1
    x_history = np.append(x_history,np.reshape(x,(2,1)),axis=1)
    f_history = np.append(f_history,f(x))

# vizualization
delta = 0.01
x = np.arange(-3.0, 3.0, delta)
y = np.arange(-2.0, 2.0, delta)
X, Y = np.meshgrid(x, y)
it = np.nditer([X, Y, None])
for x,y,z in it:
    z[...] = f(np.array([x,y]))
Z = it.operands[2]
levels = np.linspace(1.0E-06,f(x0),10)
CS = plt.contour(X, Y, Z,levels = f_history[::-1])
plt.clabel(CS, inline=1, fontsize=10)
plt.plot(x_history[0,:],x_history[1,:],':xr')
plt.show()



"""
# "worst" starting point
x0 = 1./np.diag(A)
x0 = 3.0*x0/np.linalg.norm(x0)
"""