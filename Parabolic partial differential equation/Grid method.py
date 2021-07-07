import numpy as np
from math import sqrt, sin, pi, cos, log, exp
import matplotlib.pyplot as plt
import pylab
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

aa, bb = [0,1]
T = 0.1 # in our case
N = 20 #, 10,20
M = 120 #,10,20,40,80	
h = (bb-aa)/N
tau = T/M
# tau = 0.5/(M**2)
nu = tau/h**2
sigma = 1 #, 0.5, 0 

x = [i*h for i in range((N)+1)]
# t = [i*tau*M/5 for i in range((M)+1)]
t = [i*tau for i in range((M)+1)]

def solution(x,t):
	return x**3 + t**3
	# return x + t

phi = lambda x: solution(x,0)
fi = lambda x,t: -6*x - 3*x**4 - 3*x*x + 3*t*t
a = lambda x,t: 1
b = lambda x,t: x*x + 1
c = lambda x,t: 0
alfa = lambda t: 0
alfa1 = lambda t: 0
alfa2 = lambda t: -1
beta = lambda t: 3
beta1 = lambda t: 0
beta2 = lambda t: 1


# phi = lambda x: solution(x,0)
# fi = lambda x,t: - x**2
# a = lambda x,t: 1
# b = lambda x,t: x**2 + 1
# c = lambda x,t: 0
# alfa = lambda t: 1
# alfa1 = lambda t: 0
# alfa2 = lambda t: -1
# beta = lambda t: 1 
# beta1 = lambda t: 0
# beta2 = lambda t: 1

def Lh(i,k,h,U):
	x = [i*h for i in range((N)+1)]
	t = [i*tau for i in range((M)+1)]
	L1 = (U[i+1][k-1] - 2*U[i][k-1] + U[i-1][k-1])/(h**2)
	L2 = (U[i+1][k-1] - U[i-1][k-1])/(2*h)
	L = a(x[i],t[k])*L1 + b(x[i],t[k])*L2 + c(x[i],t[k])*U[i][k-1]
	return L

def tkst(t,k,sigma):
	if sigma == 1:
		return t[k]
	if sigma == 0:
		return t[k-1]
	if sigma == 0.5:
		return t[k] - tau/2

def explicit_method():
	# x = [i*h for i in range((N)+1)]
	# t = [i*tau for i in range((M)+1)]
	U = np.zeros((N+1, M+1))
	for i in range(N+1):
		U[i][0] =  phi(x[i])
	for k in range(1,M+1):
		for i in range(1, N):
			U[i][k] = U[i][k-1] + tau*(Lh(i,k,h,U) + fi(x[i],t[k-1]))
		U[0][k] = (alfa(t[k]) + alfa2(t[k])*(4*U[1][k] - U[2][k])/(2*h)) / (alfa1(t[k]) + (3*alfa2(t[k]))/(2*h))
		U[N][k] = (beta(t[k]) + beta2(t[k])*(4*U[N-1][k] - U[N-2][k])/(2*h)) / (beta1(t[k]) + (3*beta2(t[k]))/(2*h))
	return U

def progonka(k,U):
	A = np.zeros(N+1); B = np.zeros(N+1); C = np.zeros(N+1); D = np.zeros(N+1)

	A[0] = 0
	B[0] = - (alfa1(t[k]) + alfa2(t[k])/h)
	C[0] = - alfa2(t[k])/h
	D[0] = alfa(t[k])
	B[N] = - (beta1(t[k]) + beta2(t[k])/h)
	A[N] = - beta2(t[k])/h
	C[N] = 0
	D[N] = beta(t[k])

	for i in range(1,N):
		A[i] = sigma*(a(x[i],tkst(t,k,sigma))/h**2 - b(x[i],tkst(t,k,sigma))/(2*h))
		C[i] = sigma*(a(x[i],tkst(t,k,sigma))/h**2 + b(x[i],tkst(t,k,sigma))/(2*h))
		B[i] = sigma*(2*a(x[i],tkst(t,k,sigma))/h**2 - c(x[i],tkst(t,k,sigma))) + 1/tau
		D[i] = - (1/tau)*U[i][k-1] - (1-sigma)*Lh(i,k-1,h,U) - fi(x[i],tkst(t,k,sigma))

	s = np.zeros(N+1); p = np.zeros(N+1); UU = np.zeros(N+1)
	s[0] = C[0]/B[0]; p[0] = -D[0]/B[0]
	for i in range(1,N+1):
		s[i] = C[i]/(B[i] - A[i]*s[i-1])
		p[i] = (A[i]*p[i-1]-D[i])/(B[i]-A[i]*s[i-1])

	UU[N]=p[N]
	for i in range(N-1,-1,-1):
		UU[i] = s[i]*UU[i+1] + p[i]
	return UU

def implicit_method():
	# x = [i*h for i in range((N)+1)]
	# t = [i*tau for i in range((M)+1)]
	U = np.zeros((N+1, M+1))
	for i in range(N+1):
		U[i][0] = phi(x[i])

	A = np.zeros(N+1); B = np.zeros(N+1); C = np.zeros(N+1); D = np.zeros(N+1)
	for k in range(1,M+1):
		Y = progonka(k,U)
		for i in range(N+1):
			U[i][k] = Y[i]
	return U

def plot_surface(x, t, U):
	fig = pylab.figure()
	ax = Axes3D(fig)
	x, t = np.meshgrid(x, t)
	ax.plot_surface(x, t, U,cmap = cm.jet)
	ax.set_xlabel('t')
	ax.set_ylabel('x')
	ax.set_zlabel('U(x,t)')
	pylab.show()

def main():
# 	x = [i*h for i in range((N)+1)]
# 	t = [i*tau for i in range((M)+1)]
	Q = np.zeros((N+1, M+1))
	norm1 = np.zeros((N+1, M+1))
	J_ex1 = np.zeros((N+1, M+1))
	J_ex2 = np.zeros((N+1, M+1))
	U = explicit_method()
	U2 = explicit_method()
	Y = implicit_method()
	for i in range(N+1):
		for k in range(M+1):
			Q[i][k] = solution(x[i],t[k])
			J_ex1[i][k] = abs(Q[i][k] - U[i][k])
			norm1[i][k] = abs(U[i][k] - U2[i][k])
			J_ex2[i][k] = abs(Q[i][k] - Y[i][k])

	# print(J_ex1)
	# print(J_ex2)
	maxx = 0
	Jex = 0
	for i in range(N+1):
		for k in range(M+1):
			if Jex >= maxx:
				Jex = abs(Q[i][k] - U[i][k])
				maxx = Jex
	print(maxx)

	print(h, tau)
	print('sigma = ', sigma)
	# print('N = ', N)
	# print('M = ', M)
	# plot_surface(t, x, Q)
	# plot_surface(t, x, U)
	# plot_surface(t, x, Y)

	i=0.0;k=0; res1 = np.zeros((6, 6)); res2 = np.zeros((6, 6)); res3 = np.zeros((6, 6))
	for j in range(6):
		k = 0.0
		for l in range(6):
			number1 = int(N*i)
			number2 = int(M*k*10)
			res1[l][j] = Q[number1][number2]
			res2[l][j] = U[number1][number2]
			res3[l][j] = Y[number1][number2]
			# print(Q[i][k], U[i][k], Y[i][k])
			# print('x = ',"{0:.7f}".format(x[number1]),'  ','t = ',"{0:.7f}".format(t[number2]),'  ',"{0:.7f}".format(Q[number1][number2]),'  ',"{0:.7f}".format(U[number1][number2]),'  ',"{0:.7f}".format(Y[number1][number2]))
			print(x[number1], t[number2])
			# print   (t)
			k = k + 0.02
		i = i + 0.2
#
	for i in range(6):
		print(res1[i])
		# for k in range(6):
		# 	print(res1[i][k])





main()