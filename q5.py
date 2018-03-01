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

def algorithm3_5:
    alpha0 = 0
    alpha_max = 10
    alpha = (alpha0+alpha_max)/2
    phi = f(z_list, n, x + alpha*p)
    if phi > f(z_list, n, x) + c1*

def armijoBacktracking(f, df, z_list, n, p, x): # Both conditions satisfied, using bisection
    alpha0 = 10
    alpha_max = np.inf
    alpha_min = 0
    c1 = 0.4
    c2 = 0.8
    alpha = alpha0
    counter = 0
    while True:
        counter+=1
        if not counter % 100:
            print(alpha)
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

def steepestDescent(f, df, z_list, n, xk):
    residuals = []
    residuals.append(f(z_list, n, xk))
    while f(z_list, n, xk) > 10e-4 and np.linalg.norm(df(z_list, n, xk), 2) > 10e-6:
        p = - df(z_list, n, xk)
        alpha = armijoBacktracking(f, df, z_list, n, p, xk)
        xk, xk_prev = xk + alpha * p, xk
        residuals.append(f(z_list, n, xk))
    return xk, residuals

# A Quasi-Newton Method
def BFGS(f, df, z_list, n, xk):
    residuals = []
    residuals.append(f(z_list, n, xk))
    Hk = np.identity(int(n * (n + 1) / 2) + n)
    while f(z_list, n, xk) > 10e-4 and np.linalg.norm(df(z_list, n, xk), 2) > 10e-6:
        p = - np.dot(Hk, df(z_list, n, xk))
        alpha = armijoBacktracking(f, df, z_list, n, p, xk)
        xk, xk_prev = xk + alpha * p, xk
        sk = xk - xk_prev
        yk = df(z_list, n, xk) - df(z_list, n, xk_prev)
        rho = 1 / np.dot(yk, sk)
        Hk = np.matmul(np.matmul((np.identity(int(n * (n + 1) / 2) + n) - rho * np.dot(sk, yk)), Hk),
                                (np.identity(int(n * (n + 1) / 2) + n) - rho * np.dot(yk, sk))) + rho * np.dot(sk,
                                                                                                               sk)
        residuals.append(f(z_list, n, xk))
    return xk, residuals

def SR1(f, df, z_list, n, x):
    return 0

def conjugateGradient(f, df, z_list, n, x):
    return 0

def convergencePlot(residuals):
    klist = [i for i in range(len(residuals))]
    plt.plot(klist, residuals)
    plt.xlabel("k")
    plt.ylabel("f(xk)")
    plt.legend(["Steepest Descent", "BFGS"])

if __name__ == "__main__":
    n = 2
    m = 5  # number of z points
    x = np.ones(int(n * (n + 1) / 2) + n)
    z_list = np.random.rand(m, n + 1)*10
    for i in range(m):
        z_list[i][0] = (-1)**(i % 2)
    print(z_list)

    x_steep, res_steep = steepestDescent(q4.f_model_1, q4.df_model_1, z_list, n, x)
    print("value of model 1 at end of steepest descent:", q4.f_model_1(z_list, n, x_steep))
    print("Residuals of Steepest Descent", res_steep)
    x_bfgs, res_bfgs = BFGS(q4.f_model_1, q4.df_model_1, z_list, n, x)
    print("value of model 1 at end of BFGS:", q4.f_model_1(z_list, n, x_bfgs))
    print("Residuals of BFGS", res_bfgs)

    convergencePlot(res_steep)
    convergencePlot(res_bfgs)
    plt.show()
