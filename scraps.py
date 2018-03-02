def steepestDescent(f, df, z_list, n, xk):
    residuals = np.zeros(300)
    counter = 0
    residuals[counter] = f(z_list, n, xk)
    while f(z_list, n, xk) > 0 and np.linalg.norm(df(z_list, n, xk), 2) > 10e-4:
        p = - df(z_list, n, xk)
        alpha = armijoBacktracking(f, df, z_list, n, p, xk)
        xk, xk_prev = xk + alpha * p, xk
        counter += 1
        residuals[counter] = f(z_list, n, xk)
    return xk, residuals

# Armijo er visst det samme som backtracking
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
            if np.isinf(alpha_max):
                alpha = 2*alpha
            else:
                alpha = (alpha_min + alpha_max)/2
        else:
            return alpha

def zoom(f, df, z_list, n, p, x, alpha_lo, alpha_hi, c1, c2, ):
    while True:
        # bisection interpolation to find trial between alphas
        alpha = (alpha_lo + alpha_hi) / 2
        phi = f(z_list, n, x + alpha * p)
        if phi > f(z_list, n, x) + c1 * alpha * df(z_list, n, x) or phi >= f(z_list, n, x + alpha_lo * p):
            alpha_hi = alpha
        else:
            phi_der = np.dot(df(z_list, n, x + alpha * p), p)
            if abs(phi_der) >= -c2 * df(z_list, n, x):
                return alpha
            if phi_der*(alpha_hi-alpha_lo) >= 0:
                alpha_hi = alpha_lo
            alpha_lo = alpha

def algorithm3_5(f, df, z_list, n, p, x):
    c1 = 0.6
    c2 = 0.6
    alpha_max = 1
    alpha_prev = 0
    alpha = alpha_max/2
    phi = f(z_list, n, x + alpha*p)
    print("phi", phi)
    i = 1
    while True:
        if phi > f(z_list, n, x) + c1 * alpha * df(z_list, n, x) or (phi >= f(z_list, n, x + alpha_prev * p) and i > 1):
            return zoom(f, df, z_list, n, p, x, alpha_prev, alpha, c1, c2)
        phi_der = np.dot(df(z_list, n, x + alpha * p), p)
        if abs(phi_der) >= -c2 * df(z_list, n, x):
            return alpha
        if phi_der >= 0:
            return zoom(f, df, z_list, n, p, x, alpha, alpha_prev, c1, c2)
        alpha, alpha_prev = (alpha + alpha_max) / 2, alpha
        i += 1

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

    area = 2
    z_list = np.random.uniform(-area, area, (m, n + 1))
    for i in range(m):
        z_list[i][0] = (-1)**(i % 2)
    z_list = Task3.classify_by_ellipse(m,n,area)
    print(z_list)

# A Trust Region Method
# Assumption vi bør spørre om: Det er greit å starte med B = Identity som førsteutkast til B?
def SR1(f, df, z_list, n, xk):
    B = np.identity(n)
    delta = 0.2
    epsilon = 10e-3
    eta = 10e-4
    r = 0.5
    while f(z_list, n, xk) > 10e-4 and np.linalg.norm(df(z_list, n, xk), 2) > 10e-6:
        #Compute sk by solving subproblem
        yk = df(z_list, n, xk + sk) - df(z_list, n, xk)
        ared = f(z_list, n, xk) - f(z_list, n, xk + sk)
        pred = -(np.dot(df(z_list, n, xk), sk) + 0.5*np.dot(sk, np.matmul(B, sk)))
        if ared/pred > eta:
            xk, xk_prev = xk + sk, xk
        else:
            xk, xk_prev = xk, xk

