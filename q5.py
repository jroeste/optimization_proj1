import numpy as np
import matplotlib.pyplot as plt
import time
import q4 as q4

# Optimization Algorithms

def backtrackingLinesearch(f, df, z_list, n, p, x): # Satisfies sufficient decrease, doesn't care about curvature condition
    alpha0 = 1
    rho = 0.5
    c1 = 0.4
    alpha = alpha0
    f0 = f(z_list, n, x)
    while True:  #in the worst case it takes a small step
        if f(z_list, n, x + alpha*p) <= f0 + c1 * alpha * np.dot(df(z_list, n, x), p):
            return alpha
        else:
            alpha = rho * alpha

def armijoBacktracking(f, df, z_list, n, p, x): # Both conditions satisfied, using bisection
    alpha0 = 1
    alpha_max = np.inf
    alpha_min = 0
    c1 = 0.4
    c2 = 0.8
    alpha = alpha0
    while True:
        # Suff decr. Sjekker om vi har gått for langt
        if f(z_list, n, x + alpha*p) >= f(z_list, n, x) + c1 * alpha * np.dot(df(z_list, n, x), p):
            alpha_max = alpha
            alpha = (alpha_min + alpha_max)/2
        # Curvature. Sjekker om vi har gått for kort
        elif np.dot(df(z_list, n, x + alpha*p), p) < c2 * np.dot(df(z_list, n, x), p):
            alpha_min = alpha
            if alpha_max == np.inf:
                alpha = 2*alpha
            else:
                alpha = (alpha_min + alpha_max)/2
        else:
            return alpha

def steepestDescent(f, df, z_list, n, x):
    xk_prev = x
    xk = x
    p = - df(z_list, n, xk) # descent direction
    alpha = armijoBacktracking(f, df, z_list, n, p, xk) # step length
    xk = xk_prev + alpha * p
    while f(z_list, n, xk) > 0 and np.linalg.norm(df(z_list, n, x), 2) > 10e-4:
        p = - df(z_list, n, xk)
        alpha = armijoBacktracking(f, df, z_list, n, p, xk)
        xk, xk_prev = xk + alpha * p, xk
    return xk

# A Quasi-Newton Method
def BFGS(f, df, z_list, n, xk):
    Hk = np.identity(int(n * (n + 1) / 2) + n)
    while f(z_list, n, xk) > 0 and np.linalg.norm(df(z_list, n, x), 2) > 10e-4:
        p = - np.matmul(Hk, df(z_list, n, xk))
        alpha = armijoBacktracking(f, df, z_list, n, p, xk)
        xk, xk_prev = xk + alpha * p, xk
        print("\nxk", xk)
        print("f(xk)", f(z_list, n, xk))
        sk = xk - xk_prev
        yk = df(z_list, n, xk) - df(z_list, n, xk_prev)
        rho = 1 / np.dot(yk, sk)
        Hk = np.matmul(np.matmul((np.identity(int(n * (n + 1) / 2) + n) - rho * np.dot(sk, yk)), Hk),
                                (np.identity(int(n * (n + 1) / 2) + n) - rho * np.dot(yk, sk))) + rho * np.dot(sk,
                                                                                                               sk)
    return xk

def SR1(f, df, z_list, n, x):
    return 0

if __name__ == "__main__":
    n = 2
    dim = int(n * (n + 1) / 2) + n
    m = 4  # number of z points
    x = np.ones(dim)
    z_list = np.zeros((m, n + 1))
    for i in range(m):
        z_list[i] = np.ones(n + 1) * i
        if abs(i-1.5) > 1:
            z_list[i][0] = -1
        else:
            z_list[i][0] = 1
    # m=3 gir to w = -1 og en w = 1
    print("n", n)
    print("xo", x)
    result_x = steepestDescent(q4.f_model_1, q4.df_model_1, z_list, n, x)
    print("value of model 1 at end of steepest descent:", q4.f_model_1(z_list, n, result_x))
    result_x = BFGS(q4.f_model_1, q4.df_model_1, z_list, n, x)
    print("value of model 1 at end of BFGS:", q4.f_model_1(z_list, n, result_x))
