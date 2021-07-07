import matplotlib.pyplot as plt
import math
import numpy as np
from numpy.linalg import norm, inv
import math
import functools
import fractions
from numpy import array,identity,diagonal

print('Enter the size of matrix')
s = int(input())

a_const = 3.0
b_const = 1.0

A = np.zeros((s,s))

for i in range(s):
	A[i,i] = a_const

for i in range(s-1):
	A[i,i+1] = ((i+1)*(i+2)/(s*(s+1))) * b_const

for i in range(s-1):
	A[i+1,i] = A[i,i+1]

for i in range(s):
	for k in range(s):
		if abs(i-k) > 1:
			A[i,k] = 2/((i+1)**2+(k+1)**2)
# print (A)

def PowerIterMetod(A):

	X=np.array([1]*s)
	X.resize(s,1)

	x=np.dot(A,X)	
	x1=float(x[0][0])
	d = norm(x,ord=2) - norm(X,ord=2)
	n=0

	while abs(d) > 0.00000001:   # 10^8
		X=x
		x=np.dot(A,X)
		x1=float(x[0][0])
		n=n+1
		res = x
		x=x/x1
		d = norm(X,ord=2) - norm(x,ord=2)
	print ('num of itt',n)
	return res

w , v = np.linalg.eig(A) 

lambdamax = PowerIterMetod(A)
maxl = lambdamax[0][0]
print ('calcul lambdamax: ', maxl)

maxW = max(w,key=abs)	
print ('theory lambdamax: ', maxW)

print('----------------------------')

ainv = inv(A) 
lambdamin = PowerIterMetod(ainv)
lambdamin = lambdamin**(-1)
minl = lambdamin[0][0]
print ('calcul lambdamin: ', minl) 

minW = min(w,key=abs)
print ('theory lambdamin:  ', minW)

def maxElem(a): # Find largest off-diag. element a[k,l]
    n = len(a)
    aMax = 0.0
    # for i in range(n-1):
    #     for j in range(i+1,n):
    #         if abs(a[i,j]) >= aMax:
    for i in range(n):
        for j in range(n):
            if (i!=j) and abs(a[i,j]) >= aMax:
                aMax = abs(a[i,j])
                k = i
                l = j
    return aMax, k,l

def rotate(a,p,k,l): # Rotate to make a[k,l] = 0
    n = len(a)
    aDiff = a[l,l] - a[k,k]
    if abs(a[k,l]) < abs(aDiff)*1.0e-36: t = a[k,l]/aDiff
    else:
        phi = aDiff/(2.0*a[k,l])
        t = 1.0/(abs(phi) + math.sqrt(phi**2 + 1.0))
        if phi < 0.0: t = -t
    c = 1.0/math.sqrt(t**2 + 1.0); s = t*c
    tau = s/(1.0 + c)
    temp = a[k,l]
    a[k,l] = 0.0
    a[k,k] = a[k,k] - t*temp
    a[l,l] = a[l,l] + t*temp
    for i in range(k):      # Case of i < k
        temp = a[i,k]
        a[i,k] = temp - s*(a[i,l] + tau*temp)
        a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
    for i in range(k+1,l):  # Case of k < i < l
        temp = a[k,i]
        a[k,i] = temp - s*(a[i,l] + tau*a[k,i])
        a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
    for i in range(l+1,n):  # Case of i > l
        temp = a[k,i]
        a[k,i] = temp - s*(a[l,i] + tau*temp)
        a[l,i] = a[l,i] + s*(temp - tau*a[l,i])
    for i in range(n):      # Update transformation matrix
        temp = p[i,k]
        p[i,k] = temp - s*(p[i,l] + tau*p[i,k])
        p[i,l] = p[i,l] + s*(temp - tau*p[i,l])

def Jacobi(A):
	n=len(A)
	eps = 0.00000000000001
	h = 0
	p = np.zeros((n,n))
	for i in range(n):
		p[i,i] = 1
	aMax = 1
	while (aMax>eps):
		aMax, k,l = maxElem(A)
		rotate(A,p,k,l)
		h = h +1
	return diagonal(A), h, A

U, h , A=Jacobi(A)
print('----------------------------')
print('num of itt ', h)
print('all lambda:', U)
print(A)


def Rot(A,c,s):
    n=len(A)
    V = np.zeros((n,n))
    aMax, p,q = maxElem(A)
    for i in range(n):
        V[i,i] = 1
    V[p,q] = -s
    V[q,p] = s
    V[q,q] = c
    V[p,p] = c
    print(V)
    return V

def MatrixC(A):
    n=len(A)
    C = np.zeros((n,n))
    aMax, j,k = maxElem(A)
    print(aMax,j,k)

    if A[j,j] == A[k,k]:
        tet = math.pi/4
        s = math.sin(tet)
        c = math.cos(tet)
        d = math.sqrt((A[j,j] - A[k,k])**2 + 4*A[j,k]**2)
    else:
        d = math.sqrt((A[j,j] - A[k,k])**2 + 4*A[j,k]**2)
        c = math.sqrt((1 + abs(A[j,j] - A[k,k])/d)/2)
        s = np.sign(A[j,k]*(A[j,j] - A[k,k]))*math.sqrt((1 - abs(A[j,j] - A[k,k])/d)/2)

    print(d,c,s)
    for m in range(n):
        C[m,j] = c*A[m,j] + s*A[m,k]
        C[j,m] = c*A[m,j] + s*A[m,k]
        C[m,k] = - s*A[m,j] + c*A[m,k]
        C[k,m] = - s*A[m,j] + c*A[m,k]
        for l in range(n):
            C[m,l] = A[m,l]
    C[j,j] = c**2*A[j,j] - 2*s*c*A[j,k] + s**2*A[k,k]
    C[k,k] = s**2*A[j,j] + 2*s*c*A[j,k] + c**2*A[k,k]
    # C[j,j] = (A[j,j] + A[k,k])/2 + np.sign(A[j,j] - A[k,k])*d/2
    # C[k,k] = (A[j,j] + A[k,k])/2 - np.sign(A[j,j] - A[k,k])*d/2
    C[j,k] = 0
    C[k,j] = 0
    return C 

def Jacobi(A):
    n = len(A)
    eps = 10**(-12)
    h = 0
    tA=1

    while (tA>eps):
        # aMax, j,k = maxElem(A)

        # if A[j,j] == A[k,k]:
        #     tet = math.pi/4
        #     s = math.sin(tet)
        #     c = math.cos(tet)
        # else:
        #     tau = (A[j,j]-A[k,k])/(2*A[j,k]) 
        #     t = np.sign(tau)*1/(abs(tau)+math.sqrt(1+tau**2)) 
        #     c = 1/math.sqrt(1+t**2) 
        #     s = c*t

        # print('c= ',c,'s= ',s)
        # V = Rot(A,c,s)
        # VT = np.transpose(V)
        # A = np.transpose(A)

        # A = np.dot(A,V)
        # A = np.dot(VT,A) 
        # A = Matrix(A,c,s)

        A = MatrixC(A)
        tA = diag(A)
        
        # if h % el == 0:
        #     print('tA = ', tA)
            # print('eps = ', eps)
        h = h +1
    return diagonal(A), h