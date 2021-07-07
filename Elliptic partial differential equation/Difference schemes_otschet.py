import numpy as np
from math import sqrt, sin, pi, cos, log, exp, acos
from scipy.integrate import dblquad, quad
import matplotlib.pyplot as plt
import pylab
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

ax, lx = [0,1]
ay, ly = [0,1]
n = 5
m = 10
hx = (lx-ax)/n
hy = (ly-ay)/m

eps = 0.1
p = lambda x,y: log(2 + x)
q = lambda x,y: log(2 + y)
mu = lambda x,y: sin(pi*x)*cos(pi*y)
solution_func = lambda x,y: sin(pi*x)*cos(pi*y)

x = [i*hx for i in range(n+1)]
y = [i*hy for i in range(m+1)]

solution = np.zeros((n+1, m+1))
for i in range(n+1):
	for j in range(m+1):
		solution[i][j] = solution_func(x[i],y[j])

c1, c2, d1, d2 = 1,0,1,0
for i in range (1,n):
	for j in range(1,m):
		if  p(x[i],y[j]) < c1:
			c1 = p(x[i],y[j])
		if p(x[i],y[j]) > c2:
			c2 = p(x[i],y[j])
		if q(x[i],y[j]) < d1:
			d1 = q(x[i],y[j])		
		if q(x[i],y[j]) > d2:
			d2 = q(x[i],y[j])

delta = (c1*4*sin(pi*hx/(2*lx))**2)/hx**2 + (d1*4*sin(pi*hy/(2*ly))**2)/hy**2
Delta = c2*4/hx**2 + d2*4/hy**2
nu = delta/Delta
omega = 2/(sqrt(delta*Delta))
gamma1 = delta/(2 + 2*sqrt(nu))
gamma2 = delta/(4*sqrt(nu))
k1 = omega/hx**2
k2 = omega/hy**2
zeta = delta/Delta
# zeta = gamma1/gamma2
ro = (1 - zeta)/(1 + zeta)

def Lu(u):
	L1 = p(x[i] + hx/2, y[j])*(u[i+1][j] - u[i][j])/hx**2
	L2 = p(x[i] - hx/2, y[j])*(u[i][j] - u[i-1][j])/hx**2
	L3 = q(x[i], y[j] + hy/2)*(u[i][j+1] - u[i][j])/hy**2
	L4 = q(x[i], y[j] - hy/2)*(u[i][j] - u[i][j-1])/hy**2
	L = L1 - L2 + L3 - L4
	return L

def dirichle(u):
	for i in range(n+1):
		u[i][0] = mu(x[i], 0)
		u[i][m] = mu(x[i], ly)
	for j in range(1, m):
		u[0][j] = mu(0,y[j])
		u[n][j] = mu(lx,y[j])
	return u

def f(x,y):
	a = (sin(pi*x)*(sin(pi*y) + pi*(y + 2)*log(y+2)*cos(pi*y)))/(y+2)
	b = (cos(pi*y)*(cos(pi*x) - pi*(x + 2)*log(x+2)*sin(pi*x)))/(x+2)
	fi = pi*(a - b)
	return fi

def simple_itt(u):
	U = np.zeros((n+1, m+1))
	U = dirichle(U)
	for i in range(1,n):
		for j in range(1,m):
			u1 = (p(x[i] - hx/2, y[j]) * u[i-1][j])/hx**2
			u2 = (p(x[i] + hx/2, y[j]) * u[i+1][j])/hx**2
			u3 = (q(x[i], y[j] - hy/2) * u[i][j-1])/hy**2
			u4 = (q(x[i], y[j] + hy/2) * u[i][j+1])/hy**2
			u5 = p(x[i] - hx/2, y[j])/hx**2
			u6 = p(x[i] + hx/2, y[j])/hx**2
			u7 = q(x[i], y[j] - hy/2)/hy**2
			u8 = q(x[i], y[j] + hy/2)/hy**2
			U[i][j] = (u1 + u2 + u3 + u4 + f(x[i],y[j]))/(u5 + u6 + u7 + u8)
	return U

def optimal(u):
	tau = 2/(delta + Delta)
	U = np.zeros((n+1, m+1))
	U = dirichle(U)
	for i in range(1,n):
		for j in range(1,m):
			u1 = p(x[i] + hx/2, y[j])*(u[i+1][j] - u[i][j])/hx**2
			u2 = p(x[i] - hx/2, y[j])*(u[i][j] - u[i-1][j])/hx**2
			u3 = q(x[i], y[j] + hy/2)*(u[i][j+1] - u[i][j])/hy**2
			u4 = q(x[i], y[j] - hy/2)*(u[i][j] - u[i][j-1])/hy**2
			U[i][j] = u[i][j] + tau*(u1 - u2 + u3 - u4 + f(x[i],y[j]))
	return U

def zeidel(u):
	U = np.zeros((n+1, m+1))
	U = dirichle(U)
	for i in range(1,n):
		for j in range(1,m):
			u1 = (p(x[i] - hx/2, y[j]) * U[i-1][j])/hx**2
			u2 = (p(x[i] + hx/2, y[j]) * u[i+1][j])/hx**2
			u3 = (q(x[i], y[j] - hy/2) * U[i][j-1])/hy**2
			u4 = (q(x[i], y[j] + hy/2) * u[i][j+1])/hy**2
			u5 = p(x[i] - hx/2, y[j])/hx**2
			u6 = p(x[i] + hx/2, y[j])/hx**2
			u7 = q(x[i], y[j] - hy/2)/hy**2
			u8 = q(x[i], y[j] + hy/2)/hy**2
			U[i][j] = (u1 + u2 + u3 + u4 + f(x[i],y[j]))/(u5 + u6 + u7 + u8)
	return U

def altern_triang(Uk):
	tau = 2/(gamma1 + gamma2)
	U = np.zeros((n+1, m+1)); W = np.zeros((n+1, m+1))
	WW = np.zeros((n+1, m+1)); F = np.zeros((n+1, m+1))
	u = np.zeros((n+1, m+1))

	u = np.copy(Uk)
	u = dirichle(u)

	for i in range(1,n):
		for j in range(1,m):
			F[i][j] = Lu(u) + f(x[i],y[j])
			uu1 = k1*p(x[i] - hx/2, y[j])*WW[i-1][j]
			uu2 = k2*q(x[i], y[j] - hy/2)*WW[i][j-1]
			uu3 = k1*p(x[i] - hx/2, y[j])
			uu4 = k2*q(x[i], y[j] - hy/2)
			WW[i][j] = (uu1 + uu2 + F[i][j])/(1 + uu3 + uu4) 

	for i in range(n-1,0,-1):
		for j in range(m-1,0,-1):	
			u1 = k1*p(x[i] + hx/2, y[j])*W[i+1][j]
			u2 = k2*q(x[i], y[j] + hy/2)*W[i][j+1]
			u3 = k1*p(x[i] + hx/2, y[j])
			u4 = k2*q(x[i], y[j] + hy/2)
			W[i][j] = (u1 + u2 + WW[i][j])/(1 + u3 + u4)

	for i in range(n+1):
		for j in range(m+1):
			U[i][j] = Uk[i][j] + tau*W[i][j]
	return U

def plot_surface(x, t, U):
	fig = pylab.figure()
	ax = Axes3D(fig)
	x, t = np.meshgrid(x, t)
	ax.plot_surface(x, t, U,cmap = cm.jet)
	ax.set_xlabel('y')
	ax.set_ylabel('x')
	ax.set_zlabel('U(x,y)')
	pylab.show()

def main(eps):
	U = np.zeros((n+1, m+1)); Y = np.zeros((n+1, m+1))
	Q = np.zeros((n+1, m+1)); O = np.zeros((n+1, m+1))
	J = np.zeros((n+1, m+1)); u = np.zeros((n+1, m+1))

	all_eps = 1; k = 150
	# while all_eps > eps:
	for l in range(k):
		U = simple_itt(U)
		Y = optimal(Y)
		Q = zeidel(Q)
		O = altern_triang(O)

		eps1, eps2, eps3, eps4, eps5 = 1,1,1,1,1
		for i in range (1,n):
			for j in range(1,m):
				if solution[i][j] < eps1:
					eps1 = solution[i][j]
				if U[i][j] < eps2:
					eps2 = U[i][j]
				if Y[i][j] < eps3:
					eps3 = Y[i][j]				
				if Q[i][j] < eps4:
					eps4 = Q[i][j]				
				if O[i][j] < eps5:
					eps5 = O[i][j]

		e1 = abs(eps1-eps2)
		e2 = abs(eps1-eps3)
		e3 = abs(eps1-eps4)
		e4 = abs(eps1-eps5)

		all_eps = max(e1,e2,e3)
		# k += 1
		# if k%10 == 0:
		# 	plot_surface(y, x, u)

		# print("%.6f" % e1, 'simple_itt')
		# print("%.6f" % e2, 'optimal')
		# print("%.6f" % e3, 'zeidel')
		# print("%.6f" % e4, 'altern_triang')
		# print('k = ', k)
		# print("%.0f" % k, '   ',"%.6f" % e1, '   ',"%.6f" % e2, '   ',"%.6f" % e3, '   ',"%.6f" % e4)

	print(y)
	for i in range(n+1):
		a=x[i]
		b=U[i]
		print("%.1f" % x[i],  solution[i])
	print('----------------------------------------------')
	print(y)
	for i in range(n+1):
		a=x[i]
		b=U[i]
		print("%.1f" % x[i],  U[i])
	print('----------------------------------------------')

	print(y)
	for i in range(n+1):
		a=x[i]
		b=U[i]
		print("%.1f" % x[i],  Y[i])
	print('----------------------------------------------')

	print(y)
	for i in range(n+1):
		a=x[i]
		b=U[i]
		print("%.1f" % x[i],  Q[i])

	# plot_surface(y, x, solution)
	# plot_surface(y, x, U)
	# plot_surface(y, x, Y)
	# plot_surface(y, x, Q)
	# plot_surface(y, x, O)

main(eps)