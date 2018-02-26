import numpy as np
import matplotlib.pyplot as plt
import q4 as q4

# Two conditions to guarantee convergence:
#   Armijo condition = sufficient decrease condition
#   Curvature condition

# f trenger z_list, A, c        df trenger z_list, n, A, b, c

def backtrackingLinesearch(f, df, z_list, n, p, x): # Satisfies sufficient decrease, doesn't care about curvature condition
    alpha0 = 1
    rho = 0.5
    c1 = 0.5
    alpha = alpha0
    A0, c0 = q4.construct_A_and_C(n, x)
    f0 = f(z_list, A0, c0)
    while True:
        A, c = q4.construct_A_and_C(n, x + alpha*p)
        b = 0
        # Her oppretter vi matrise på nytt hver gang... Må vi det?
        #if f(x + alpha*p) <= f(z_list, x) + c1*alpha*df(x)*p:
        print("f", f(z_list, A, c))
        print("f0 + c1 * alpha * df(z_list, n, A, b, c) * p", f0 + c1 * alpha * df(z_list, n, A, b, c) * p
        if f(z_list, A, c) <= f0 + c1 * alpha * df(z_list, n, A, b, c) * p:
            return alpha
        else:
            alpha = rho * alpha

# kan evt lage en core linesearch method som motor inne i funksjoner steepest descent, newton etc for å finne p

def steepestDescent(f, df, z_list, n, x):
    xk_prev = x
    A, c = q4.construct_A_and_C(n, x)
    b = 0
    # p is the search direction. Different approaches for finding it yields different algorithms
    gradient = df(z_list, n, A, b, c)
    p = - gradient/np.linalg.norm(gradient, 2)
    alpha = backtrackingLinesearch(f, df, z_list, n, p, x) # finding step length
    xk = xk_prev + alpha * gradient * p
    while np.linalg.norm(xk-xk_prev, 2) > 10e-4:
        A, c = q4.construct_A_and_C(n, x)
        gradient = df(z_list, n, A, b, c)
        p = - gradient / np.linalg.norm(gradient, 2)
        alpha = backtrackingLinesearch(f, df, z_list, n, p, x)  # finding step length
        xk, xk_prev = xk + alpha * gradient * p, xk
    return xk

def quasiNewton():
    p = 0 # something something
    return 0

if __name__ == "__main__":
    # Følgende er blåkopi fra q4 26.02
    # dimensions must be n = int(k*(k+1)/2) such that k is an integer
    n = 6
    dim = int(n*(n+1)/2) + n
    m = 3  # number of z points
    x = np.ones(dim)
    z_list = np.zeros((m, n + 1))
    for i in range(m):
        z_list[i] = np.ones(n + 1) * i
        if i < int(m / 2):
            z_list[i][0] = -1
        else:
            z_list[i][0] = 1
    print("n", n)
    print("dim of x", dim)
    # m=3 gir to w = -1 og en w = 1
    A, c = q4.construct_A_and_C(n, x)

    # optimization algorithms require f and x0. They find direction p locally in their own way
    print("value of model 1 at x0:", q4.f_model_1(z_list, A, c))
    steepestDescent(q4.f_model_1, q4.df_model_1, z_list, n, x)