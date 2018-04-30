import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import q4 as q4
import q5 as q5


def eval_func_model_1_2D(X,Y,A,c):
    return A[0][0]*(X-c[0])**2 + 2*A[0][1]*(X-c[0])*(Y-c[1]) + A[1][1]*(Y-c[1])**2

def eval_func_model_2_2D(X,Y,A,b):
    return A[0][0]*X**2+2*A[0][1]*X*Y+A[1][1]*Y**2+b[0]*X+b[1]*Y

def classify_by_ellipse(m,n,area):
    a00=np.random.uniform(0,2)
    a11=np.random.uniform(0,2)
    minval=min(a00,a11)
    a01=np.random.uniform(0,minval)
    A = [[a00,a01], [a01, a11]]  #symmetric, positive definite A
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


def classify_by_rectangle(m,n,area,min,max):
    rec = [-area / np.random.uniform(min, max)
        , area / np.random.uniform(min, max)
        , -area / np.random.uniform(min, max)
        , area / np.random.uniform(min, max)]
    z = np.zeros((m, n + 1))
    '''Perform classification:'''
    for i in range(m):
        z[i] = np.random.uniform(-area, area, n + 1)
        x = z[i][1]
        y = z[i][2]
        if rec[0] < x < rec[1] and rec[2]< y < rec[3]:
            z[i][0] = 1
        else:
            z[i][0] = -1
    return z

def classify_misclassification(m,n,area,prob):
    z_list=classify_by_ellipse(m,n,area)
    for i in range(m):
        a=np.random.uniform()
        if a<prob:
            z_list[i][0]*=-1
    return z_list

def plot_rectangle_and_points(m,n,area,rec):
    rec_left = rec[0]
    rec_right = rec[1]
    rec_lower = rec[2]
    rec_upper = rec[3]

    z=classify_by_rectangle(m,n,area,min,max)
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

def plot_dataset_2d(X,Y,Z,col):
    CS = plt.contour(X, Y, Z, [1],linestyles=col)
    plt.clabel(CS, inline=1, fontsize=10)

def plot_z_points(z,title):
    for i in range(m):
        if z[i][0]<0:
            col='green'
        else:
            col='red'
        plt.plot(z[i][1], z[i][2], 'o', color=col)
    plt.title(title)

def make_ellipse(A,c,area,func):
    delta = 0.01
    x = np.arange(-area*1.01, 1.01*area+delta, delta)
    y = np.arange(-area*1.01, 1.01*area+delta, delta)
    X, Y = np.meshgrid(x, y)
    Z=func(X, Y, A, c)
    return X,Y,Z


def create_easy_z_list(m,n):
    z_list = np.zeros((m,n+1))
    z_list[0][0]=1
    z_list[0][1]=0
    z_list[0][2]=0
    z_list[1][0]=-1
    z_list[1][1]=1
    z_list[1][2]=0
    print(z_list)
    return z_list

if __name__=='__main__':
    '''Constants:'''
    m=50
    n=2
    area=2.0
    x_length=int(n*(n+1)/2)+n
    prob=0.05
    min_rec,max_rec=1,4
    '''Initials'''

    x_initial = np.zeros(x_length)
    x_initial[0], x_initial[2] = 1, 1
    x_initial=np.ones(x_length)

    '''Create dataset'''
    title_1B = "Classify by ellipse - BFGS - m = "+str(m)
    title_1F = "Classify by ellipse - FR - m = "+str(m)
    title_2B = "Classify by rectangle - BFGS - m = "+str(m)
    title_2F = "Classify by rectangle - FR - m = "+str(m)
    title_3B = "Classify with error 5 % - BFGS - m = "+str(m)
    title_3F = "Classify with error 5 % - FR - m = "+str(m)

    #z_list = classify_by_ellipse(m, n, area)
    z_list = classify_by_rectangle(m, n, area, min_rec,max_rec)
    #z_list=classify_misclassification(m,n,area,prob)


    #z_list=create_easy_z_list(m,n)

    #Model 1:
    x_vector_model_1 = q5.BFGS(q4.f_model_1, q4.df_model_1, z_list, n, x_initial)

    A1,c1=q4.construct_A_and_C(n,x_vector_model_1[0])
    X1,Y1,Z1=make_ellipse(A1,c1,area,eval_func_model_1_2D)
    plot_dataset_2d(X1, Y1, Z1,'solid')

    #Model 2:
    x_vector_model_2 = q5.BFGS(q4.f_model_2, q4.df_model_2, z_list, n, x_initial)

    A2,c2=q4.construct_A_and_C(n,x_vector_model_2[0])
    X2,Y2,Z2=make_ellipse(A2,c2,area,eval_func_model_2_2D)
    plot_dataset_2d(X2,Y2,Z2,'dashed')
    plot_z_points(z_list,title_2B)
    plt.grid()
    plt.show()


