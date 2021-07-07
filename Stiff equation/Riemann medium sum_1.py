import matplotlib.pyplot as plt
import math
import numpy as np

print('Vvedi A<0')
A = int(input())
print('Vvedi n')
n = int(input())
# A = -50
# n=20
C= 7 # 5, 10
d=C/abs(A)

print('Значение А = ',A)
print('Значение n = ',n)

Y=np.zeros(2*n+1)
X=np.zeros(2*n+1)
Z=np.zeros(2*n+1)

Y[0] = 10
Z[0] = -1

def solve(x0,xend,n, n1, n2,Y,Z):
	h=(xend - x0)/n
	eps = 1e-9

	Czn1=1-A*h/2-25*A*h*h/(60-18*h)
	Czn=1
	Cyn=-25*h/(30-9*h)
	Cx=Cyn/2

	for i in range(n1, n2+1):
		znew = 1 
		while (abs(znew - Z[i]) > eps):
			Z[i] = znew 
			sl1 = Z[i-1] * Czn
			sl2 = Y[i-1] * Cyn 
			sl3 = ( x0 + h*(i-0.5) )**2 / ( (x0 + h*(i-0.5))**3 + 1 ) * Cx
			sl4 = - h*math.sqrt( Z[i-1]**2 + Z[i-1]+1 )/2
			znew = (sl1 + sl2 + sl3 + sl4) / Czn1
		Z[i] = znew
		sl1 = Y[i-1]
		sl3 = Z[i] * (-A*h/2) ############ -Ah
		sl4 = (x0 + h*(i-0.5))**2/((x0 + h*(i-0.5))**3 + 1) * h
		Y[i] = (sl1 + sl3 + sl4) /(1 - 3/10*h) ##### 1- 3/5*h

h=d/n
for i in range(1,n+1):
	X[i] = X[i-1] + h
	# print(i, X[i],h)
solve(0,d,n,1,2*n, Y, Z)

h=(1-d)/n
for i in range(n+1,2*n+1):
	X[i]=X[i-1] + h
	# print(i, X[i],h)
solve(d,1,n,n+1,2*n-1,Y,Z)	

YY=np.zeros(2*n+2)
ZZ=np.zeros(2*n+2)
XX=np.zeros(2*n+2)
YY[0]=Y[0]
ZZ[0]=Z[0]
XX[0]=X[0]

h=d/n
x0=0
for i in range(0,n+1):
	XX[i]=( x0 + h*(i-0.5) )**2 / ( (x0 + h*(i-0.5))**3 + 1 )
	YY[i+1]=YY[i]+h*(3/5*Y[i]-A*Z[i]+XX[i])
	ZZ[i+1]=ZZ[i]+h*(-5/3*Y[i]+A*Z[i]-math.sqrt( Z[i]**2 + Z[i]+1 ))

h=(1-d)/n
x0=d
for i in range(n+1,2*n+1):
	XX[i]=( x0 + h*(i-0.5) )**2 / ( (x0 + h*(i-0.5))**3 + 1 )
	YY[i+1]=YY[i]+h*(3/5*Y[i]-A*Z[i]+XX[i])
	ZZ[i+1]=ZZ[i]+h*(-5/3*Y[i]+A*Z[i]-math.sqrt( Z[i]**2 + Z[i]+1 ))
print(YY[0], ZZ[0])
print('i        X[i]          Y[i]            Z[i]  ')
for i in range(0,2*n+1):
	print("{0:.0f}".format(i),'  ',"{0:.8f}".format(X[i]),'  ',"{0:.8f}".format(YY[i]),'    ' ,"{0:.8f}".format(ZZ[i]))

# y = YY
# # x = XX
# z = ZZ
# plt.plot(y,z)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Решение краевой задачи')
# plt.savefig('sol80.png')