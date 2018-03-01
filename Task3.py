import numpy as np
import matplotlib.pyplot as plt
import q4 as q4
import matplotlib.cm as cm
import matplotlib.mlab as mlab

def function_evaluate_model_1(X,Y,A,c):
    return A[0][0]*(X-c[0])**2 + 2*A[0][1]*(X-c[0])*(Y-c[1]) + A[1][1]*(Y-c[1])**2

def classify_by_ellipse(m,n,):
    #På ein eller anna passande måte laga random A,C som gir random ellipse innenfor gitte rammer
    A = [[0.4, 0.4], [0.4, 0.6]]
    c = np.random.uniform(-1, 1, n)
    #Opretta random z-punkter innenfor passande område
    #Bruk function_evaluate til å klassifisere
    #Returner datasett

#def classify_by_rectangle():
    #som by_ellipse, finn eit passande tilfeldig rektangel og bruk istedenfor

#def classify_misclassification():
    #bruk ellipse/rectangle og opprett sannsynlighet for missklassifisering'

def plot_dataset_2d(X,Y,Z):
    #Plot datasett sammen med ellipse fra optimeringsløsning
    plt.figure()
    CS = plt.contour(X, Y, Z, [1])
    #plt.clabel(CS, inline=1, fontsize=10)
    plt.show()



def make_ellipse():

    delta = 0.01
    A = [[0.6, 0.8], [0.8, 0.6]]
    c = np.random.uniform(-1, 1, 2)
    x = np.arange(-2.0, 2.01, delta)
    y = np.arange(-2.0, 2.01, delta)
    X, Y = np.meshgrid(x, y)
    Z=function_evaluate_model_1(X, Y, A, c)
    #return X,Y,Z
    plt.figure()
    CS = plt.contour(X, Y, Z,[1])
    plt.clabel(CS, inline=1, fontsize=10)
    plt.show()


def main():
    make_ellipse()
    #Lag datasett
    #Kjør optimeringsalgoritme på datasett
    #plot løsning med datasett-punkter
    #repeat for ulike metoder osv

main()