import numpy as np
import matplotlib.pyplot as plt
import time
import q4 as q4
import Task3 as Task3

def backtrackingLinesearch(f, df, z_list, n, p, x):
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

def note3algoritme(f, df, z_list, n, p, x):
    c1 = 0.5
    c2 = 0.6
    alpha = 1
    alpha_min = 0
    alpha_max = np.inf
    dfk = df(z_list, n, x)
    fk=f(z_list, n, x)
    while True:
        if f(z_list, n, x + alpha*p) > fk + c1*alpha*np.dot(dfk, p):  #No suff decr
            alpha_max = alpha
            alpha = (alpha_max + alpha_min)/2
        elif np.dot(df(z_list, n, x + alpha*p), p) < c2*np.dot(dfk, p): # No curv con
            alpha_min = alpha
            if np.isinf(alpha_max):
                alpha = 2*alpha
            else:
                alpha = (alpha_max + alpha_min) / 2
        else:
            return alpha

def steepestDescent(f, df, z_list, n, xk):
    fk = f(z_list, n, xk)
    dfk = df(z_list, n, xk)
    residuals = []
    residuals.append(fk)
    while fk > 10e-4 and np.linalg.norm(dfk, 2) > 10e-6:
        p = - dfk
        alpha = note3algoritme(f, df, z_list, n, p, xk)
        xk = xk + alpha * p
        fk=f(z_list, n, xk)
        dfk=df(z_list, n, xk)
        residuals.append(fk)
    return xk, residuals

# A Conjugate Gradient Method
def fletcherReeves(f, df, z_list, n, xk): # Nonlinear Conjugate Gradient
    maxiter=500
    residuals = []
    fk = f(z_list, n, xk)
    residuals.append(fk)
    dfk = df(z_list, n, xk)
    p = -dfk
    iter=0
    while fk >= 10e-3 and np.linalg.norm(dfk, 2) > 10e-3 and iter < maxiter:   #changed tolerance
        alpha = note3algoritme(f, df, z_list, n, p, xk)
        xk, xk_prev = xk + alpha * p, xk
        dfkplus1 = df(z_list, n, xk)
        beta_kplus1 = np.dot(dfkplus1, dfkplus1)/np.dot(dfk, dfk)
        p = -dfkplus1 + beta_kplus1*p
        dfk = dfkplus1
        fk = f(z_list, n, xk)
        residuals.append(fk)
        iter+=1
    #print(fk)
    return xk, residuals

# A Quasi-Newton Method
def BFGS(f, df, z_list, n, xk):
    residuals = []
    residuals.append(f(z_list, n, xk))
    I=np.identity(int(n * (n + 1) / 2) + n)
    Hk = I
    fk = f(z_list, n, xk)
    dfk = df(z_list, n, xk)
    maxiter=500
    iter=0
    stop=False
    while fk >= 10e-3 and np.linalg.norm(dfk, 2) > 10e-3 and stop==False:
        p = -Hk.dot(dfk)
        alpha = note3algoritme(f, df, z_list, n, p, xk)
        xk_prev=xk
        xk = xk + alpha*p
        sk = xk - xk_prev
        fk=f(z_list, n, xk)
        dfk_prev=dfk
        dfk = df(z_list, n, xk)
        yk = dfk - dfk_prev
        rho = 1 / np.dot(yk, sk)
        Hk_prev=Hk
        Hk=np.matmul(I-rho*np.outer(sk,yk),np.matmul(Hk_prev,I-rho*np.outer(yk,sk))) + rho*np.outer(sk,sk)
        residuals.append(fk)
        iter+=1
        if iter>maxiter:
            stop=True
    #print(fk)
    return xk, residuals

def construct_z_elliptic(n, m, A, c, area):
    z_list = np.random.uniform(-area, area, (m, n + 1))
    for i in range(m):
        z_list[i][0] = 1
        if q4.compute_r_i_1(z_list[i], A, c) >= 1:
            z_list[i][0] = -1
    return z_list

def plot1(n, x, m):
    A, c=q4.construct_A_and_C(n,x)
    for times in range(100):
        z_list = construct_z_elliptic(n, m, A, c, area)
        #z_list = Task3.classify_by_rectangle(m, n, area, 1, 4)
        #z_list = Task3.classify_misclassification(m, n, area, 0.05)
        x_m2, res_m2 = BFGS(q4.f_model_2, q4.df_model_2, z_list, n, x)
        klist = [i for i in range(len(res_m2))]
        plt.plot(klist, res_m2)
    plt.xlabel("k")
    plt.ylabel("f(xk)")
    plt.title("BGFS, Model 2")

def otherPlot(n, x):
    mvalues = [i for i in range(1, 20, 4)] + [i for i in range(20, 50, 10)] + [i for i in range(50, 100, 25)] + [i for i in range(150, 551, 100)]
    #mvalues = [i for i in range(1, 20, 4)]
    #print(mvalues)
    #mvalues=[i for i in range(50, 150, 20)]
    w = len(mvalues)
    print("Antall mvalues:", w)
    iterations_fr_m1 = [0] * w
    iterations_fr_m2 = [0] * w
    iterations_BFGS_m1 = [0] * w
    iterations_BFGS_m2 = [0] * w
    A, c = q4.construct_A_and_C(n,x)
    area=2.0
    #simulations=5
    simulations = 20
    for times in range(simulations):
        print("Round", times+1)
        for i in range(w):
            #z_list = construct_z_elliptic(n, mvalues[i], A, c, area)
            #z_list = Task3.classify_by_rectangle(mvalues[i], n, area, 1, 4)
            z_list = Task3.classify_misclassification(mvalues[i], n, area, 0.05)

            iterations_fr_m1[i] += len(fletcherReeves(q4.f_model_1, q4.df_model_1, z_list, n, x)[1])-1
            iterations_fr_m2[i] += len(fletcherReeves(q4.f_model_2, q4.df_model_2, z_list, n, x)[1])-1
            iterations_BFGS_m1[i] += len(BFGS(q4.f_model_1, q4.df_model_1, z_list, n, x)[1]) - 1
            iterations_BFGS_m2[i] += len(BFGS(q4.f_model_2, q4.df_model_2, z_list, n, x)[1]) - 1

            #print("functionvalue:",fletcherReeves(q4.f_model_1, q4.df_model_1, z_list, n, x)[1])
            #print("functionvalue:", fletcherReeves(q4.f_model_2, q4.df_model_2, z_list, n, x)[1])
            #print("functionvalue:", BFGS(q4.f_model_1, q4.df_model_1, z_list, n, x)[1])
            #print("functionvalue:", BFGS(q4.f_model_2, q4.df_model_2, z_list, n, x)[1])

            #print("Points:", mvalues[i], "Iterations_FL_m1:", iterations_fr_m1[i])#, "Iterations_fr_m2:", iterations_fr_m2[i])
            #print("Points:", mvalues[i], "Iterations_BFGS_m1:", iterations_BFGS_m1[i], "Iterations_BFGS_m2:", iterations_BFGS_m2[i])
            #print(i)
    iterations_fr_m1[:] = [elem/simulations for elem in iterations_fr_m1]
    iterations_fr_m2[:] = [elem / simulations for elem in iterations_fr_m2]
    iterations_BFGS_m1[:] = [elem / simulations for elem in iterations_BFGS_m1]
    iterations_BFGS_m2[:] = [elem / simulations for elem in iterations_BFGS_m2]
    plt.plot(mvalues, iterations_fr_m1)
    plt.plot(mvalues, iterations_fr_m2)
    plt.plot(mvalues, iterations_BFGS_m1)
    plt.plot(mvalues, iterations_BFGS_m2)
    plt.legend(["F-R Model 1", "F-R Model 2", "BFGS Model 1", "BFGS Model 2"])
    plt.xlabel("Points (m)")
    plt.ylabel("Iterations")
    #plt.title("Ellipse ")
    #plt.title("Rectangle")
    plt.title("Ellipse with misclassification rate 5%")

def otherPlot_BFGS():
    n = 2
    m = 100  # number of z points
    x = np.ones(int(n * (n + 1) / 2) + n)

    # Classify by ellipse
    area = 2
    A, c = q4.construct_A_and_C(n, x)
    mvalues = [i for i in range(1, 30)] # + [j for j in range(30,200,10)] + [i for i in range(200, 501, 100)]
    #print(mvalues)
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
        #print("Points:", i + 1, "Iterations_BFGS_m1:", iterations_BFGS_m1[i], "Iterations_BFGS_m2:", iterations_BFGS_m2[i])
    plt.plot(mvalues, iterations_BFGS_m1)
    plt.plot(mvalues, iterations_BFGS_m2)
    plt.legend(["Model 1", "Model 2"])
    plt.xlabel("Points (m)")
    plt.ylabel("Iterations")
    #print("plotting")
    plt.show()

if __name__ == "__main__":
    n = 2
    m = 100  # number of z points
    x = np.ones(int(n * (n + 1) / 2) + n)
    x[1] = 0
    x[3] = 0
    x[4] = 0
    area = 2
#    plot1(n, x, m)
    otherPlot(n, x)
    print("plotting ;)")
    plt.show()
