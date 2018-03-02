import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches


import q4 as q4
import q5 as q5
import matplotlib.cm as cm
import matplotlib.mlab as mlab

def eval_func_model_1_2D(X,Y,A,c):
    return A[0][0]*(X-c[0])**2 + 2*A[0][1]*(X-c[0])*(Y-c[1]) + A[1][1]*(Y-c[1])**2

def eval_func_model_2_2D(X,Y,A,b):
    return A[0][0]*X**2+2*A[0][1]*X*Y+A[1][1]*Y**2+b[0]*X+b[1]*Y

def classify_by_ellipse(m,n,area):
    '''
    - På ein eller anna passande måte laga random A,C
        som gir random ellipse innenfor gitte rammer
    - Oppretta random z-punkter innenfor passande område
    - Bruk function_evaluate til å klassifisere
    - Returner datasett'''
    A = [[1, 0.4], [0.4, 0.8]]  #symmetric, positive definite A
    c = np.random.uniform(-1, 1, n) #random vector
    z = np.zeros((m,n+1))
    for i in range(m):
        z[i]=np.random.uniform(-area,area,n+1)

    '''Perform classification:'''
    for i in range(m):
        f_value=eval_func_model_1_2D(z[i][1],z[i][2],A,c)
        if f_value>=1:  #if outside the ellipse, the weight should be -1
            z[i][0]=-1
        else:
            z[i][0]=1   #if inside the ellipse, the weight should be +1
    return z


def classify_by_rectangle(m,n,area,rec):
    z = np.zeros((m, n + 1))

    for i in range(m):
        z[i] = np.random.uniform(-area, area, n + 1)
        x = z[i][1]
        y = z[i][2]
        if rec[0] < x < rec[1] and rec[2]< y < rec[3]:
            z[i][0] = 1
        else:
            z[i][0] = -1
    return z

def classify_misclassification():
    return 0
    #bruk ellipse/rectangle og opprett sannsynlighet for missklassifisering'

def plot_rectangle_and_points(m,n,area,rec):
    rec_left = rec[0]
    rec_right = rec[1]
    rec_lower = rec[2]
    rec_upper = rec[3]

    z=classify_by_rectangle(m,n,area,rec)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    for i in range(m):
        if z[i][0]<0:
            col='green'
        else:
            col='red'
        ax1.plot(z[i][1], z[i][2], 'o', color=col)

    '''Plot rectangle'''
    ax1.add_patch(
        patches.Rectangle(
            (rec_left, rec_lower),  # (x,y)
            rec_right - rec_left,  # width
            rec_upper - rec_lower,  # height
            fill=False))
    ax1.axis([-area, area + 0.01, -area, area + 0.01])
    plt.show()

def plot_dataset_2d(X,Y,Z):
    #Plot datasett sammen med ellipse fra optimeringsløsning
    CS = plt.contour(X, Y, Z, [1])
    plt.clabel(CS, inline=1, fontsize=10)
    plt.grid()

def plot_z_points(z):
    for i in range(m):
        if z[i][0]<0:
            col='green'
        else:
            col='red'
        plt.plot(z[i][1], z[i][2], 'o', color=col)

def make_ellipse(A,c,area):
    delta = 0.1
    x = np.arange(-area, area+0.01, delta)
    y = np.arange(-area, area+0.01, delta)
    X, Y = np.meshgrid(x, y)
    Z=eval_func_model_1_2D(X, Y, A, c)
    return X,Y,Z


if __name__=='__main__':
    '''Constants:'''
    m=500
    n=2
    area=2.0
    x_length=int(n*(n+1)/2)+n

    '''Initials'''
    x_vec=np.ones(x_length) #x_vec=q5.steepest decent(f, df, z_list, n, x)
    rec = [-area/np.random.uniform(1,2)
            ,area/np.random.uniform(1,2)
            ,-area/np.random.uniform(1,2)
            ,area/np.random.uniform(1,2)]

    A = [[1, 0], [0, 4]]  # symmetric, positive definite A
    c = np.random.uniform(-1, 1, n)
    x_initial=np.ones(x_length)

    '''Create dataset'''

    z_list=classify_by_rectangle(m,n,area,rec)

    #x_vector=q5.steepestDescent(q4.f_model_1,q4.df_model_1,z_list,n,x_initial)[0]
    x_vector = q5.fletcherReeves(q4.f_model_1, q4.df_model_1, z_list, n, x_initial)[0]
    A,c=q4.construct_A_and_C(n,x_vector)
    X,Y,Z=make_ellipse(A,c,area)
    plot_dataset_2d(X,Y,Z)
    plot_z_points(z_list)
    plt.show()


    #plot_dataset_2d(X,Y,Z)
    #classify_by_ellipse(m, n, area)
    #Lag datasett
    #Kjør optimeringsalgoritme på datasett
    #plot løsning med datasett-punkter
    #repeat for ulike metoder osv
