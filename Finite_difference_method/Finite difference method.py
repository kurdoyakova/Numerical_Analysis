#! /usr/bin/env python

import numpy as np
import math
import matplotlib.pyplot as plot

def sweep_method(n):                  
    a=0
    b=1
    h=(b-a)/n
    alpha1=1
    alpha2=0
    betha1=1
    betha2=0

    x=np.zeros(n+2)
    y=np.zeros(n+2)
    p=np.zeros(n+2)
    q=np.zeros(n+2)
    f=np.zeros(n+2)

    for k in range(0,n+2):
        x[k]=a-h/2+k*h
        p[k]=1+math.sin(math.pi*x[k]/2)
        q[k]=math.log(x[k]*x[k]+4)
        f[k]=2-x[k]

    A=np.zeros(n+2)
    B=np.zeros(n+2)
    C=np.zeros(n+2)
    D=np.zeros(n+2)

    A[0]=0
    B[0]=alpha1*h/2+1
    C[0]=1-alpha1*h/2
    D[0]=alpha2*h

    for k in range(1,n+1):
        A[k]=1-p[k]*h/2
        B[k]=2-q[k]*h**2
        C[k]=1+p[k]*h/2
        D[k]=f[k]*h**2

    B[n+1]=betha1*h/2-1
    A[n+1]=-1-betha1*h/2
    C[n+1]=0
    D[n+1]=betha2*h

    # находим прогоночные коэф-ты
    alpha=np.zeros(n+2)
    beta=np.zeros(n+2)

    alpha[0]=C[0]/B[0]
    beta[0]=-D[0]/B[0]

    for i in range(1,n+2):
        alpha[i]=C[i]/(B[i]-A[i]*alpha[i-1])
        beta[i]=(A[i]*beta[i-1]-D[i])/(B[i]-A[i]*alpha[i-1])

    # методом обратной прогонки находим y[k]
    y=np.zeros(n+2)
    y[n+1]=beta[n+1]

    for i in range(n,-1,-1):
        y[i]=alpha[i]*y[i+1]+beta[i]

    return y,x,alpha,beta

def tabl(r):
    y, x, alpha, beta = sweep_method(r)
    print('   alpha   ', '      beta      ', '     y')
    for i in range(0,r+2):
        print("{0:.8f}".format(alpha[i]),'  ',"{0:.8f}".format(beta[i]),'  ',"{0:.8f}".format(y[i]),'    ' ,"{0:.8f}".format(x[i]))

    #строим график
    plot.plot(x,y)
    plot.xlabel('x')
    plot.ylabel('y')
    plot.title('Решение краевой задачи')
    plot.savefig('sol80.png')

def conv(xx,s,l,t):
    for i in range(s,l,t):
        y, x, alpha, beta = sweep_method(i)
        for k in range(0,i):
            if x[k]==xx:
                print('x=',x[k], 'y=', y[k], 'n=',i)

def choice(ch):
    if ch==1:
        print('Vvedi shislo itter')
        r = int(input())
        tabl(r)
    else:
        print('Vvedi shislo for x')
        xx = float(input())
        print('Vvedi start step')
        s = int(input())
        print('Vvedi end step')
        l = int(input())
        print('Vvedi step')
        t = int(input())
        conv(xx,s,l,t)

print('Vvedi sposob vvoda. 1= tabl, other_number=conv')
ch = int(input())
choice(ch)

# xx=0.55 #x for conv
# s=10    #start step
# l=300   #end step
# t=20    #step 
# r=80    #step for now

 #tabl(r)
# conv(xx,s,l,t)