__author__ = 'julie'
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def f_model_1(z_list,A,c):
    functionsum=0
    for i in range(len(z_list)):    #length m
        if z_list[i][0]>0:
            functionsum+=(max(np.dot((z_list[i][1:]-c),(np.matmul(A,(z_list[i][1:]-c))))-1,0))**2
        else:
            functionsum+=(max(1-np.dot((z_list[i][1:]-c),(np.matmul(A,z_list[i][1:]-c))),0))**2
    return functionsum

def f_model_2(z_list,A,b):
    functionsum=0
    for i in range(len(z_list)):    #length m
        if z_list[i][0]>0:
            functionsum+=(max(np.dot(z_list[i][1:],np.matmul(A,z_list[i][1:]))+np.dot(b,z_list[i][1:])-1,0))**2
        else:
            functionsum+=(max(1-np.dot(z_list[i][1:],np.matmul(A,z_list[i][1:]))-np.dot(b,z_list[i][1:]),0))**2
    return functionsum

def construct_A_and_C(n,x):
    A=np.zeros((n,n))
    C=x[n*(n+1)/2:]
    counter=0
    for i in range(n):
        for j in range(n-i):
            A[i][j+i]=x[n*i+counter+j]  #should change to x[counter], where counter increases with 1. 
        counter -= i
        for j in range(i):
            A[i][j]=A[j][i]
    return A,C

def df_model_1(z_list,n,A,b,c): #if model 1: b=0 and c=c, if model 2: b=b and c=0
    counter=0
    dfx=np.zeros(int(n*(n+1)/2)+n)
    for h in range(z_list):     #length m

        #find the first n*(n+1)/2 x-entries
        for i in range(n):      #length n
            for j in range(n - i):
                if i==j:
                   dfx[n*i+counter+j]+=(z_list[h][i+1]-c[i])**2
                else:
                    dfx[n*i+counter+j]+=2*(z_list[h][j+i+1]-c[j+i])*(z_list[h][i+1]-c[i])
            counter -= i

        #find the last n x-entries
        for j in range(n):
            for i in range(n):
                if i==j:
                    dfx[int(n*(n+1)/2)+j]+=-2*A[i][j]*(z_list[h][j+1]-c[j])
                else:
                    dfx[int(n*(n+1)/2)+j]+=-A[i][j]*(z_list[h][i+1]+c[i])
    return dfx



