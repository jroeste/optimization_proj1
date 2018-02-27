import numpy as np
import matplotlib.pyplot as plt
import time

# Idé: Lag HELT generelle optimeringsfunksjoner som bruker kun f og df.
# Mer forståelig, enklere å feilsøke. Hvis kjøretiden blir stor får vi endre til å sende rundt A og c etterpå.
# f får kun z_list, n, x

# Skal gradienten normeres?

#### Engines ####

def f_model_1(z_list, n, x):
    A, c = construct_A_and_C(n, x)
    functionsum=0
    for i in range(len(z_list)):    #length m
        if z_list[i][0]>0:
            functionsum+=compute_r_i_1(z_list[i],A,c,i)**2
        else:
            functionsum+=(compute_r_i_1(z_list[i],A,c,i))**2
    return functionsum

def compute_r_i_1(z_list_i,A,c,i):
    if z_list_i[0]>0:
        return max(np.dot((z_list_i[1:] - c), (np.matmul(A, (z_list_i[1:] - c)))) - 1, 0)
    else:
        return max(1-np.dot((z_list_i[1:]-c),(np.matmul(A,z_list_i[1:]-c))),0)

def construct_A_and_C(n, x):  # Her har Even vært og endret ting 26.02
        C = x[int(n * (n + 1) / 2):]
        A = np.zeros((n, n))
        index = 0
        for h in range(n):
            for j in range(n - h):
                A[h][h + j] = x[index]
                A[h + j][h] = x[index]
                index += 1
        # print("x\n", x)
        # print("A\n", A)
        # print("C contains elements in x from index", int(n*(n+1)/2), "\n", C)
        return A, C

def df_model_1(z_list, n, x):  # returns gradient, a vector
    A, c = construct_A_and_C(n, x)
    counter = 0
    dfx = np.zeros(int(n * (n + 1) / 2) + n)
    index = 0
    for i in range(len(z_list)):  # length m
        # find the first n*(n+1)/2 x-entries
        for h in range(n):  # length n
            for j in range(n - h):
                if h == j:
                    dfx[index] += 2 * compute_r_i_1(z_list[i], A, c, i) * (z_list[i][h + 1] - c[h]) ** 2
                else:
                    dfx[index] += 2 * compute_r_i_1(z_list[i], A, c, i) * (z_list[i][j + h + 1] - c[j + h]) * (
                    z_list[i][h + 1] - c[h])
            counter -= h

        # find the last n x-entries
        for j in range(n):
            for h in range(n):
                if h == j:
                    dfx[int(n * (n + 1) / 2) + j] += -2 * compute_r_i_1(z_list[i], A, c, i) * 2 * A[h][j] * (
                    z_list[i][j + 1] - c[j])  # legg til alpha
                else:
                    dfx[int(n * (n + 1) / 2) + j] += -2 * compute_r_i_1(z_list[i], A, c, i) * A[h][j] * (
                    z_list[i][h + 1] - c[h])  # legg til alpha
    return dfx

#### Optimization Algorithms ####

def backtrackingLinesearch(f, df, z_list, n, p, x): # Satisfies sufficient decrease, doesn't care about curvature condition
    alpha0 = 1
    rho = 0.5
    c1 = 0.4
    alpha = alpha0
    f0 = f(z_list, n, x)
    while True:
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
        if f(z_list, n, x + alpha*p) >= f(z_list, n, x) + c1 * alpha * np.dot(df(z_list, n, x), p):
            alpha_max = alpha
            alpha = (alpha_min + alpha_max)/2
        elif np.dot(df(z_list, n, x + alpha*p), p) < c2 * np.dot(df(z_list, n, x), p):
            alpha_min = alpha
            if alpha_max == np.inf:
                alpha = 2*alpha
            else:
                alpha = (alpha_min + alpha_max)/2
        else:
            return alpha
    #    print(alpha)
    return alpha

def steepestDescent(f, df, z_list, n, x):
    xk_prev = x
    p = - df(z_list, n, x) # descent direction
    alpha = armijoBacktracking(f, df, z_list, n, p, x) # step length
    xk = xk_prev + alpha * p
    print("xk", xk)
    print("f(xk)", f(z_list, n, xk))
    while abs(f(z_list, n, xk) - f(z_list, n, xk_prev)) > 10e-3:
        print("\nNy runde")
        A, c = construct_A_and_C(n, xk)
        print("A", A)
        print("c", c)
        print("f(xk)", f(z_list, n, xk))
        p = - df(z_list, n, xk)
        alpha = armijoBacktracking(f, df, z_list, n, p, xk)
        xk, xk_prev = xk_prev + alpha * p, xk
    print("xk", xk)
    return xk

if __name__ == "__main__":
    n = 2
    dim = int(n*(n+1)/2) + n
    m = 4  # number of z points
    x = np.ones(dim)
    z_list = np.zeros((m, n + 1))
    for i in range(m):
        z_list[i] = np.ones(n + 1) * i
        if abs(i-1.5) > 1:
            z_list[i][0] = -1
        else:
            z_list[i][0] = 1
    print("n", n)
    print("dim of x", dim)
    # m=3 gir to w = -1 og en w = 1
    A, c = construct_A_and_C(n, x)
    print(z_list)
    print(A)
    print(c)
    # optimization algorithms require f and x0. They find direction p locally in their own way
    print("xo", x)
    print("f(x0):", f_model_1(z_list, n, x))
    result_x = steepestDescent(f_model_1, df_model_1, z_list, n, x)
    print("value of model 1 at end of steepest descent:", f_model_1(z_list, n, result_x))
