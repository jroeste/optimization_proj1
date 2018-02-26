__author__ = 'julie'
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def f_model_1(z_list,A,c):
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

#fix this
def f_model_2(z_list,A,b):
    functionsum=0
    for i in range(len(z_list)):    #length m
        if z_list[i][0]>0:
            functionsum+=(max(np.dot(z_list[i][1:],np.matmul(A,z_list[i][1:]))+np.dot(b,z_list[i][1:])-1,0))**2
        else:
            functionsum+=(max(1-np.dot(z_list[i][1:],np.matmul(A,z_list[i][1:]))-np.dot(b,z_list[i][1:]),0))**2
    return functionsum

def construct_A_and_C(n,x): # Her har Even vært og endret ting 26.02
    C=x[int(n*(n+1)/2):]
    A=np.zeros((n,n))
    index=0
    for h in range(n):
        for j in range(n-h):
            A[h][h + j] = x[index]
            A[h + j][h] = x[index]
            index+=1
    print("x\n", x)
    print("A\n", A)
    print("C contains elements in x from index", int(n*(n+1)/2), "\n", C)
    return A, C

def df_model_1(z_list,n,A,b,c): #if model 1: b=0 and c=c, if model 2: b=b and c=0
    counter=0
    dfx=np.zeros(int(n*(n+1)/2)+n)
    index=0
    for i in range(z_list):     #length m
        #find the first n*(n+1)/2 x-entries
        for h in range(n):      #length n
            for j in range(n - h):
                if h==j:
                    dfx[index] += 2*compute_r_i_1(z_list[i],A,c,i)*(z_list[i][h + 1] - c[h]) ** 2
                else:
                    dfx[index] += 2 * compute_r_i_1(z_list[i],A,c,i)*(z_list[i][j + h + 1] - c[j + h]) * (z_list[i][h + 1] - c[h])
            counter -= h

        #find the last n x-entries
        for j in range(n):
            for h in range(n):
                if h==j:
                    dfx[int(n*(n+1)/2)+j]+=-2*compute_r_i_1(z_list[i],A,c,i)*2*A[h][j]*(z_list[i][j+1]-c[j]) #legg til alpha
                else:
                    dfx[int(n*(n+1)/2)+j]+=-2*compute_r_i_1(z_list[i],A,c,i)*A[h][j]*(z_list[i][h+1]-c[h])   #legg til alpha
    return dfx

if __name__ == "__main__":  # Her har Even vært og endret ting 26.02
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
    print("z_list\n",z_list)
    print("dim of x\n", dim)
    # m=3 gir to w = -1 og en w = 1
    A, c = construct_A_and_C(n, x)
    print("value of model 1:", f_model_1(z_list, A, c))
