import numpy as np
from math import sqrt, sin, pi, cos, log, exp, acos, log
from scipy.integrate import dblquad, quad
import matplotlib.pyplot as plt

# aa, bb = [0,1]
# n = 15
# h = (bb-aa)/n
alfa = 0.0001
number = 5   # Сколько графиков и альф хотите


a=np.float32(0); b=np.float32(1); n = 60 # n-количество узлов. Значит промежутков n-1
h=(b-a)/(n-1);
#задаём узлы коллокации:
x=np.zeros(n)
for i in range(1,n):
	x[i]=x[i-1]+h

# x = [i*h for i in range(n+1)]

K = lambda x,t: 1/(3/2+x+t)**2
f = lambda t: log(3*(2*t + 3)/(2*t + 5))/(1 + t) - log(2)/(t + 3/2) - log(3/2)/(t + 5/2)

phi = lambda i,t: cos(i*acos(t))
right = lambda t,a: K(a,t)*f(t)
left = lambda t,y,i,x: K(x,y)*K(y,t)*phi(i,t)
func = lambda t,i,x: K(x,t)*phi(i,t)

def solution(alfa):

	CC = np.zeros((n, n))
	F = np.zeros(n)
	for j in range(n):
		F[j]= quad(right,0,1, args=(x[j]))[0]
		for i in range(n):
			CC[j][i]=alfa*phi(i,x[j]) + dblquad(left,0,1,0,1, args=(i,x[j]))[0]
	# C = np.linalg.solve(CC, F)
	C = np.linalg.inv(CC).dot(F)


	eps = np.zeros(n)
	S = np.zeros(n)
	for j in range(n):
		for i in range(n):
			S[j] = S[j] + C[i]*quad(func,0,1,args=(i,x[j]))[0]
		eps[j] = abs(f(x[j]) - S[j])
		# print('error for i =', j, 'steps =  ',eps)
	print('errors = ',eps)
	U = np.zeros(n)
	for j in range(n):
		for i in range(n):
			U[j] = U[j] + C[i]*phi(i,x[j])
	return U

def pltplot(l,x, alfa):   # Рисуут графики с шагом alfa*10 
	al = alfa*(10**l)
	print('Alfa = ', al)
	A = solution(al)
	plt.plot(x, A, label = 'alfa = ' + str(al))
	print("_____________________________________")

def main():
	fig, ax = plt.subplots()
	for i in range(number):
		pltplot(i,x,alfa)

	ax.set(xlabel='x', ylabel='U')
	ax.legend()
	ax.grid()

	fig.savefig("test.png")
	plt.show()

main()