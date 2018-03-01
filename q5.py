import numpy as np
import matplotlib.pyplot as plt
import time
import q4 as q4
import Task3 as Task3

# Denne finner alpha
def note3algoritme(f, df, z_list, n, p, x):
    c1 = 0.5
    c2 = 0.6
    alpha = 1
    alpha_min = 0
    alpha_max = np.inf
    while True: # Eneste forskjell er > i stedet for >=
        if f(z_list, n, x + alpha*p) > f(z_list, n, x) + c1*alpha*np.dot(df(z_list, n, x), p):  #No suff decr
            alpha_max = alpha
            alpha = (alpha_max + alpha_min)/2
        elif np.dot(df(z_list, n, x + alpha*p), p) < c2*np.dot(df(z_list, n, x), p): # No curv con
            alpha_min = alpha
            if np.isinf(alpha_max):
                alpha = 2*alpha
            else:
                alpha = (alpha_max + alpha_min) / 2
        else:
            return alpha

def steepestDescent(f, df, z_list, n, xk):
    residuals = []
    residuals.append(f(z_list, n, xk))
    while f(z_list, n, xk) > 10e-4 and np.linalg.norm(df(z_list, n, xk), 2) > 10e-6:
        p = - df(z_list, n, xk)
        alpha = note3algoritme(f, df, z_list, n, p, xk)
        xk, xk_prev = xk + alpha * p, xk
        residuals.append(f(z_list, n, xk))
    return xk, residuals

# A Conjugate Gradient Method
def fletcherReeves(f, df, z_list, n, xk): # Nonlinear Conjugate Gradient
    residuals = []
    residuals.append(f(z_list, n, xk))
    gradf_prev = df(z_list, n, xk)
    p = -df(z_list, n, xk)
    while f(z_list, n, xk) > 10e-4 and np.linalg.norm(df(z_list, n, xk), 2) > 10e-6:
        alpha = note3algoritme(f, df, z_list, n, p, xk)
        xk, xk_prev = xk + alpha * p, xk
        gradf_kplus1 = df(z_list, n, xk)
        beta_kplus1 = np.dot(gradf_kplus1, gradf_kplus1)/np.dot(gradf_prev, gradf_prev)
        p = - gradf_kplus1 + beta_kplus1*p
        gradf_prev = gradf_kplus1
        residuals.append(f(z_list, n, xk))
    return xk, residuals

# A Quasi-Newton Method
def BFGS(f, df, z_list, n, xk):
    residuals = []
    residuals.append(f(z_list, n, xk))
    Hk = np.identity(int(n * (n + 1) / 2) + n)
    while f(z_list, n, xk) > 10e-4 and np.linalg.norm(df(z_list, n, xk), 2) > 10e-6:
        p = - np.dot(Hk, df(z_list, n, xk))
        alpha = note3algoritme(f, df, z_list, n, p, xk)
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
    B = np.identity(n)
    delta = 0.2
    epsilon = 10e-3
    eta = 10e-4
    r = 0.5
    while f(z_list, n, xk) > 10e-4 and np.linalg.norm(df(z_list, n, xk), 2) > 10e-6:


def convergenceFR():
    x_fr_1, res_fr_1 = fletcherReeves(q4.f_model_1, q4.df_model_1, z_list, n, x)
    klist = [i for i in range(len(res_fr_1))]
    plt.loglog(klist, res_fr_1)
    x_fr_2, res_fr_2 = fletcherReeves(q4.f_model_2, q4.df_model_2, z_list, n, x)
    klist = [i for i in range(len(res_fr_2))]
    plt.loglog(klist, res_fr_2)
    plt.xlabel("k")
    plt.ylabel("f(xk)")
    plt.legend(["Model 1", "Model 2"])
#    plt.legend(["Fletcher Reeves", "Steepest Descent", "BFGS"])

if __name__ == "__main__":
    n = 6
    m = 100  # number of z points
    x = np.ones(int(n * (n + 1) / 2) + n)

# Her bruker jeg Julie sin classify by ellipse for å lage punkter, men i vilkårlig dimensjon
    area = 2
    A, c=q4.construct_A_and_C(n,x)
    z_list = np.random.uniform(-area, area, (m, n + 1))
    for i in range(m):
        z_list[i][0] = 1
        if q4.compute_r_i_1(z_list[i],A,c) >= 1:
            z_list[i][0] = -1
    print(z_list)



#    x_steep, res_steep = steepestDescent(q4.f_model_1, q4.df_model_1, z_list, n, x)
#    print("value of model 1 at end of steepest descent:", q4.f_model_1(z_list, n, x_steep))
#    print("Residuals of Steepest Descent", res_steep[-5:])
#    x_fr, res_fr = fletcherReeves(q4.f_model_1, q4.df_model_1, z_list, n, x)
#    print("value of model 1 at end of fletcher reeves:", q4.f_model_1(z_list, n, x_fr))
#    print("Residuals of Fletcher reeves", res_fr[-5:])
#    x_bfgs, res_bfgs = BFGS(q4.f_model_1, q4.df_model_1, z_list, n, x)
#    print("value of model 1 at end of BFGS:", q4.f_model_1(z_list, n, x_bfgs))
#    print("Residuals of BFGS", res_bfgs[-5:])

#    convergencePlot(res_bfgs)
    convergenceFR()
    plt.show()
