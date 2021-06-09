#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plot

a=0
b=1
n=20
h=(b-a)/n
x=[];y=[]
p=[];q=[];f=[]
alpha1=2
alpha2=1
betha1=0
betha2=0
for k in range(0,n+2):
    x.append(a-h/2+k*h)
    p.append((1-x[k])/(1+x[k]))
    q.append(math.exp(x[k]/2))
    f.append(1+x[k]**2)

x=np.array(x)
p=np.array(p)
q=np.array(q)
f=np.array(f)


A=[];B=[];C=[];D=[]
A.append(0)
B.append(alpha1*h/2+1)
C.append(1-alpha1*h/2)
D.append(alpha2*h)

for k in range(1,n+1):
    A.append(1-p[k]*h/2)
    B.append(2-q[k]*h**2)
    C.append(1+p[k]*h/2)
    D.append(f[k]*h**2)

B.append(betha1*h/2-1)
A.append(-1-betha1*h/2)
C.append(0)
D.append(betha2*h)
A=np.array(A)
B=np.array(B)
C=np.array(C)
D=np.array(D)

# находим прогоночные коэф-ты
s=[]
t=[]
s.append(C[0]/B[0])
t.append(-D[0]/B[0])
for i in range(1,n+2):
    s.append(C[i]/(B[i]-A[i]*s[i-1]))
    t.append((A[i]*t[i-1]-D[i])/(B[i]-A[i]*s[i-1]))
s=np.array(s)
t=np.array(t)

# методом обратной прогонки находим y_k

y=np.zeros(n+2)
y[n+1]=t[n+1]

for i in range(n,-1,-1):
    y[i]=s[i]*y[i+1]+t[i]
for i in range(0,n+2):
    print("{0:.7f}".format(s[i]),'  ',"{0:.7f}".format(t[i]),'  ',"{0:.7f}".format(y[i]))

plot.plot(x,y)
plot.xlabel('x')
plot.ylabel('y')
plot.title('Решение краевой задачи')
plot.savefig('approx.png')
