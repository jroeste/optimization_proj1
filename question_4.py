__author__ = 'julie'
# -*- coding: utf-8 -*-

import numpy as np

def f(x):
    # f = sin(x0) + exp(x1*x2*...)
    x0 = x[0]
    x1 = x[1:]
    return np.sin(x0)+np.exp(np.prod(x1))

def df(x):
    # grad f = [cos(x0)], exp(x1*x2*...)*x2*x3*..., ...]
    g = np.zeros_like(x)
    x0 = x[0]
    x1 = x[1:]
    g[0] = np.cos(x0)
    for i in range(0,len(x1)):
        g[i+1] = np.exp(np.prod(x1)) * np.prod(x1[np.arange(len(x1)) != i])
    return g

if __name__=='__main__':
    N = 10
    # generate random point and direction
    x = np.random.randn(N)
    p = np.random.randn(N)
    f0= f(x)
    g = df(x).dot(p)
    print(df(x))
    print(p)
    print(g)
    # compare directional derivative with finite differences
    for ep in 10.0**np.arange(-1,-20,-1):
        g_app = (f(x+ep*p)-f0)/ep
        error = abs(g_app-g)/abs(g)
        print('ep = %e, error = %e' % (ep,error))

