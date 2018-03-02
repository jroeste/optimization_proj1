import numpy as np
import matplotlib.pyplot as plt
import time
import q4 as q4

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

# Denne finner alpha
def note3algoritme(f, df, z_list, n, p, x):
    c1 = 0.5
    c2 = 0.6
    alpha = 1
    alpha_min = 0
    alpha_max = np.inf
    counter = 0
    while True: # Eneste forskjell er > i stedet for >=
        dfk = df(z_list, n, x)
        if f(z_list, n, x + alpha*p) > f(z_list, n, x) + c1*alpha*np.dot(dfk, p):  #No suff decr
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
#        if counter >100:
#            print("Step selection brukte over 100 iterasjoner. Returnerer Backtracking.")
#            return backtrackingLinesearch(f, df, z_list, n, p, x)

def steepestDescent(f, df, z_list, n, xk):
    residuals = []
    residuals.append(f(z_list, n, xk))
    while f(z_list, n, xk) > 10e-4 and np.linalg.norm(df(z_list, n, xk), 2) > 10e-2:
        p = - df(z_list, n, xk)
        alpha = note3algoritme(f, df, z_list, n, p, xk)
        xk, xk_prev = xk + alpha * p, xk
        residuals.append(f(z_list, n, xk))
    return xk, residuals

# A Conjugate Gradient Method
def fletcherReeves(f, df, z_list, n, xk): # Nonlinear Conjugate Gradient
    residuals = []
    fk = f(z_list, n, xk)
    residuals.append(fk)
    dfk = df(z_list, n, xk)
    p = -dfk
    counter = 0
    while fk > 10e-4 and np.linalg.norm(dfk, 2) > 10e-6:
        start = time.time()
        alpha = note3algoritme(f, df, z_list, n, p, xk) # Tar denne lang tid?
        #print("note3algo brukte", time.time() - start, "sek")
        xk, xk_prev = xk + alpha * p, xk
        dfkplus1 = df(z_list, n, xk)
        beta_kplus1 = np.dot(dfkplus1, dfkplus1)/np.dot(dfk, dfk)
        p = - dfkplus1 + beta_kplus1*p
        dfk = dfkplus1
        fk = f(z_list, n, xk)
        residuals.append(fk)
        counter += 1
        if counter >= 100:
            return
    return xk, residuals

# A Quasi-Newton Method
def BFGS(f, df, z_list, n, xk):
    residuals = []
    residuals.append(f(z_list, n, xk))
    Hk = np.identity(int(n * (n + 1) / 2) + n)
    fk = f(z_list, n, xk)
    dfk = df(z_list, n, xk)
    while fk > 10e-4 and np.linalg.norm(dfk, 2) > 10e-6:
        p = - np.dot(Hk, dfk)
        alpha = note3algoritme(f, df, z_list, n, p, xk)
        xk, xk_prev = xk + alpha * p, xk
        fk = f(z_list, n, xk)
        sk = xk - xk_prev
        yk = df(z_list, n, xk) - df(z_list, n, xk_prev)
        rho = 1 / np.dot(yk, sk)
        Hk = np.matmul(np.matmul((np.identity(int(n * (n + 1) / 2) + n) - rho * np.dot(sk, yk)), Hk),
                                (np.identity(int(n * (n + 1) / 2) + n) - rho * np.dot(yk, sk))) + rho * np.dot(sk,
                                                                                                               sk)
        residuals.append(f(z_list, n, xk))
    print("ferdig med BFGS")
    return xk, residuals

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
    plt.title("Fletcher Reeves, m = 100")
#    plt.legend(["Fletcher Reeves", "Steepest Descent", "BFGS"])

def otherPlot():
    n = 2
    m = 100  # number of z points
    x = np.ones(int(n * (n + 1) / 2) + n)

    # Classify by ellipse
    area = 2
    A, c = q4.construct_A_and_C(n, x)
    mvalues = [i for i in range(1, 30)] # + [j for j in range(30,200,10)] + [i for i in range(200, 501, 100)]
    print(mvalues)
    w = len(mvalues)
    z_list = np.random.uniform(-area, area, (w, m, n + 1))
    for j in range(w):
        for i in range(m):
            z_list[j][i][0] = 1
            if q4.compute_r_i_1(z_list[j][i], A, c) >= 1:
                z_list[j][i][0] = -1
    iterations_fr_m1 = [0] * w
    iterations_fr_m2 = [0] * w
    iterations_BFGS_m1 = [0] * w
    iterations_BFGS_m2 = [0] * w
    for i in range(w):
        iterations_fr_m1[i] = len(fletcherReeves(q4.f_model_1, q4.df_model_1, z_list[i], n, x)[1])-1
        iterations_fr_m2[i] = len(fletcherReeves(q4.f_model_2, q4.df_model_2, z_list[i], n, x)[1])-1
        print("Points:", i+1, "Iterations_fr_m1:", iterations_fr_m1[i], "Iterations_fr_m2:", iterations_fr_m2[i])
        iterations_BFGS_m1[i] = len(BFGS(q4.f_model_1, q4.df_model_1, z_list[i], n, x)[1]) - 1
        iterations_BFGS_m2[i] = len(BFGS(q4.f_model_2, q4.df_model_2, z_list[i], n, x)[1]) - 1
        print("Points:", i + 1, "Iterations_BFGS_m1:", iterations_BFGS_m1[i], "Iterations_BFGS_m2:", iterations_BFGS_m2[i])
    plt.plot(mvalues, iterations_fr_m1)
    plt.plot(mvalues, iterations_fr_m2)
    plt.legend(["Model 1", "Model 2"])
    plt.xlabel("Points (m)")
    plt.ylabel("Iterations")
    print("plotting")
    plt.show()

def otherPlot_BFGS():
    n = 2
    m = 100  # number of z points
    x = np.ones(int(n * (n + 1) / 2) + n)

    # Classify by ellipse
    area = 2
    A, c = q4.construct_A_and_C(n, x)
    mvalues = [i for i in range(1, 30)] # + [j for j in range(30,200,10)] + [i for i in range(200, 501, 100)]
    print(mvalues)
    w = len(mvalues)
    z_list = np.random.uniform(-area, area, (w, m, n + 1))
    for j in range(w):
        for i in range(m):
            z_list[j][i][0] = 1
            if q4.compute_r_i_1(z_list[j][i], A, c) >= 1:
                z_list[j][i][0] = -1
    iterations_BFGS_m1 = [0] * w
    iterations_BFGS_m2 = [0] * w
    for i in range(w):
        iterations_BFGS_m1[i] = len(BFGS(q4.f_model_1, q4.df_model_1, z_list[i], n, x)[1]) - 1
        iterations_BFGS_m2[i] = len(BFGS(q4.f_model_2, q4.df_model_2, z_list[i], n, x)[1]) - 1
        print("Points:", i + 1, "Iterations_BFGS_m1:", iterations_BFGS_m1[i], "Iterations_BFGS_m2:", iterations_BFGS_m2[i])
    plt.plot(mvalues, iterations_BFGS_m1)
    plt.plot(mvalues, iterations_BFGS_m2)
    plt.legend(["Model 1", "Model 2"])
    plt.xlabel("Points (m)")
    plt.ylabel("Iterations")
    print("plotting")
    plt.show()


if __name__ == "__main__":
    n = 2
    m = 100  # number of z points
    x = np.ones(int(n * (n + 1) / 2) + n)
    x[1] = 0
    x[3] = 0
    x[4] = 0

# Her bruker jeg Julie sin classify by ellipse for å lage punkter, men i vilkårlig dimensjon
    area = 2
    A, c=q4.construct_A_and_C(n,x)
    z_list = np.random.uniform(-area, area, (m, n + 1))
    for i in range(m):
        z_list[i][0] = 1
        if q4.compute_r_i_1(z_list[i],A,c) >= 1:
            z_list[i][0] = -1
    #print(z_list)

    # convergenceFR()
    otherPlot_BFGS()
