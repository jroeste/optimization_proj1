__author__ = 'julie'
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import time



def f_model_1(z_list,n,x):
    functionsum=0
    A, c=construct_A_and_C(n,x)
    for i in range(len(z_list)):    #length m
        functionsum+=(compute_r_i_1(z_list[i],A,c))**2
    return functionsum


def compute_r_i_1(z_list_i,A,c):
    if z_list_i[0]>0:
        return max([np.dot((z_list_i[1:] - c), (np.matmul(A, (z_list_i[1:] - c)))) - 1, 0])
    else:
        return max([1-np.dot((z_list_i[1:]-c),(np.matmul(A,(z_list_i[1:]-c)))),0])


def f_model_2(z_list,n,x):
    A,b=construct_A_and_C(n,x)  #endre på construct??
    functionsum=0
    for i in range(len(z_list)):    #length m
        functionsum+=compute_r_i_2(z_list[i],A,b)**2
    return functionsum

def compute_r_i_2(z_list_i,A,b):
    if z_list_i[0]>0:
        return max([np.dot(z_list_i[1:],np.matmul(A,z_list_i[1:]))+np.dot(b,z_list_i[1:])-1,0])
    else:
        return max([1-np.dot(z_list_i[1:],np.matmul(A,z_list_i[1:]))-np.dot(b,z_list_i[1:]),0])


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
        if ri==0:
            continue
        else:
            #find the first n*(n+1)/2 x-entries
            for h in range(n):      #length n
                for j in range(h,n):
                    if h==j:
                        dfx[index] += z_list[i][0]*2*ri*(z_list[i][h + 1] - c[h]) ** 2
                    else:
                        dfx[index] += z_list[i][0]*4 * ri*(z_list[i][j + 1] - c[j]) * (z_list[i][h + 1] - c[h])
                    index+=1

            #find the last n x-entries for  or c's
            for h in range(n):
                for j in range(n):
                    dfx[int(n * (n + 1) / 2) + h] += -z_list[i][0] * 4 * ri * A[j][h] * (z_list[i][j + 1] - c[j])
    return dfx


def df_model_2(z_list,n,x):
    A,b = construct_A_and_C(n,x)
    dfx=np.zeros(int(n*(n+1)/2)+n)
    for i in range(len(z_list)):     #length m
        index = 0
        ri=compute_r_i_2(z_list[i], A, b)
        if ri==0:
            continue
        else:
            #find the first n*(n+1)/2 x-entries
            for h in range(n):      #length n
                for j in range(h,n):
                    if h==j:
                        dfx[index] += z_list[i][0]*2*ri*z_list[i][h + 1] ** 2
                    else:
                        dfx[index] += z_list[i][0]*4* ri*(z_list[i][j + 1]) * (z_list[i][h + 1])
                    index+=1

            #find the last n x-entries
            for h in range(n):
                dfx[int(n * (n + 1) / 2) + h] += z_list[i][0]*2*ri*z_list[i][h+1]
    return dfx

def test_derivatives(m,n,N,funcval,d_funcval): #Ferdig
    # generate random point and direction
    x = np.random.randn(N)
    z = np.random.randn(m,n+1)
    for i in range(m):
        if i%2==0:
            z[i][0]=1
        else:
            z[i][0]=-1
    p = np.random.randn(N)
    f0 = funcval(z, n, x)
    g = d_funcval(z,n,x).dot(p)
    #compare directional derivative with finite differences
    for ep in 10.0 ** np.arange(-1, -13, -1):
        g_app = (funcval(z,n,x + ep * p) - f0) / ep #z_list,A,c
        error = abs(g_app - g) / abs(g)
        print('ep = %e, error = %e' % (ep, error))


if __name__ == "__main__":  # Her har Even vært og endret ting 26.02
    n = 3
    N = int(n*(n+1)/2+n)
    m=10
    test_derivatives(m,n,N,f_model_2,df_model_2)

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


