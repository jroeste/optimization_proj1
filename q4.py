__author__ = 'julie'
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#should be good
def f_model_1(z_list,n,x):
    functionsum=0
    A, c=construct_A_and_C(n,x)
    for i in range(len(z_list)):    #length m
        functionsum+=(compute_r_i_1(z_list[i],A,c))**2
    return functionsum

#should be correct
def compute_r_i_1(z_list_i,A,c):
    if z_list_i[0]>0:
        return max(np.dot((z_list_i[1:] - c), (np.matmul(A, (z_list_i[1:] - c)))) - 1, 0)
    else:
        return max(1-np.dot((z_list_i[1:]-c),(np.matmul(A,(z_list_i[1:]-c)))),0)

#Need to make a compute_r_i_2 function
def f_model_2(z_list,A,b):
    functionsum=0
    for i in range(len(z_list)):    #length m
        if z_list[i][0]>0:
            functionsum+=(max(np.dot(z_list[i][1:],np.matmul(A,z_list[i][1:]))+np.dot(b,z_list[i][1:])-1,0))**2
        else:
            functionsum+=(max(1-np.dot(z_list[i][1:],np.matmul(A,z_list[i][1:]))-np.dot(b,z_list[i][1:]),0))**2
    return functionsum

#should be correct
def construct_A_and_C(n,x): # Her har Even vært og endret ting 26.02
    C=x[int(n*(n+1)/2):]
    A=np.zeros((n,n))
    index=0
    for h in range(n):
        for j in range(h,n):
            A[h][j] = x[index]
            A[j][h] = x[index]
            index+=1
    return A, C


def df_model_1(z_list,n,x):
    A,c = construct_A_and_C(n,x)
    dfx=np.zeros(int(n*(n+1)/2)+n)
    for i in range(len(z_list)):     #length m
        index = 0
        ri=compute_r_i_1(z_list[i], A, c)

        #find the first n*(n+1)/2 x-entries
        for h in range(n):      #length n
            for j in range(h,n):
                if h==j:
                    dfx[index] += z_list[i][0]*2*ri*(z_list[i][h + 1] - c[h]) ** 2
                else:
                    dfx[index] += z_list[i][0]*4 * ri*(z_list[i][j + 1] - c[j]) * (z_list[i][h + 1] - c[h])
                index+=1

        #find the last n x-entries
        for j in range(n):
            for h in range(n):
                if h==j:
                    dfx[int(n*(n+1)/2)+j]+=-z_list[i][0]*4*ri*A[h][j]*(z_list[i][j+1]-c[j]) #legg til alpha
                else:
                    dfx[int(n*(n+1)/2)+j]+=-z_list[i][0]*4*ri*A[h][j]*(z_list[i][h+1]-c[h])   #legg til alpha
    return dfx

def test_derivatives(m,n,N): #Ferdig
    # generate random point and direction
    x = np.random.randn(N)
    z = np.random.randn(m,n+1)
    for i in range(m):
        if i%2==0:
            z[i][0]=1
        else:
            z[i][0]=-1
    p = np.random.randn(N)
    f0 = f_model_1(z, n, x)
    g = df_model_1(z,n,x).dot(p)
    #compare directional derivative with finite differences
    for ep in 10.0 ** np.arange(-1, -13, -1):
        g_app = (f_model_1(z,n,x + ep * p) - f0) / ep #z_list,A,c
        error = abs(g_app - g) / abs(g)
        print('ep = %e, error = %e' % (ep, error))

if __name__ == "__main__":  # Her har Even vært og endret ting 26.02
    n = 3
    N = int(n*(n+1)/2+n)
    m=10
    test_derivatives(m,n,N)

    #Noke du vil ha her Even?

    # dimensions must be n = int(k*(k+1)/2) such that k is an integer
    #n = 6
    #dim = int(n*(n+1)/2) + n
    #m = 3  # number of z points
    #x = np.ones(dim)
    #z_list = np.zeros((m, n + 1))
    #for i in range(m):
    #    z_list[i] = np.ones(n + 1) * i
    #    if i < int(m / 2):
    #        z_list[i][0] = -1
    #    else:
    #        z_list[i][0] = 1
    #print("n", n)
    #print("z_list\n",z_list)
    #print("dim of x", dim)
    ## m=3 gir to w = -1 og en w = 1
    #A, c = construct_A_and_C(n, x)
    #print("value of model 1:", f_model_1(z_list, A, c))


