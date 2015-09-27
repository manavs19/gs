import sys
from math import *
from random import *
import numpy as np
import scipy
import scipy.linalg
import numpy.random as npr
import matplotlib.pyplot as plt

# def getStandardNormal(size):
# 	X = []
# 	mean=0.0
# 	lastMean=0.0
# 	std=1.0
# 	for i in range(1,size+1):
# 		x = gauss(mean,std)
# 		X.append(x)
# 		lastMean=mean
# 		mean = mean + (x-mean)/(i+1)
# 		std = (1.0-1.0/i)*std*std + (i+1)*((mean-lastMean)**2)
# 		std=sqrt(std)

# 	return X

def getStandardNormal():
	return gauss(0.0, 1.0)

def calculateXT(X0, sigma, T, z):
	a = sigma*sqrt(T)*z - sigma*sigma*T/2.0
	return X0*exp(a)

def question1(tc, line):
	numIter = 1000000
	T = float(line[0])
	# optionType = int(line[1])
	X0 = float(line[2])
	K = float(line[3])
	sigma = float(line[4])

	# T=2.0
	# X0=0.3
	# K=0.02
	# sigma=0.001

	# Z = getStandardNormal(numIter)
	total=0.0
	for i in range(0,numIter):
		z = getStandardNormal()
		XT = calculateXT(X0, sigma, T, z)
		if XT>=K:
			total+=XT-K
	total/=numIter

	print str(tc)+",1"+","+'{0:.4f}'.format(total)#total

def question2(tc, line):
	numIter = 1000000
	rho = float(line[0])
	T = float(line[1])
	# optionType = int(line[2])
	X10 = float(line[3])
	X20 = float(line[4])
	K = float(line[5])
	sigma1 = float(line[6])
	sigma2 = float(line[7])

	# rho=0.01
	# T=0.1
	# X10=0.35
	# X20=0.08
	# K=0.01
	# sigma1=0.1
	# sigma2=0.1

	SIGMA = np.array([[1.0, rho],
					 [rho, 1.0]])
	C = scipy.linalg.cholesky(SIGMA)
	R = np.random.randn(2,numIter)
	Z = (C.transpose()).dot(R)
	# print np.cov(Z)

	total=0.0
	for i in range(0, numIter):
		X1T = calculateXT(X10, sigma1, T, Z[0][i])
		X2T = calculateXT(X20, sigma2, T, Z[1][i])
		XT = X1T * X2T
		if XT>=K:
			total+=XT-K
	total/=numIter

	print str(tc)+",2"+","+'{0:.4f}'.format(total)#str(total)

def getTotal(rho, numIter, X0, sigma, T, K):
	# print sigma
	SIGMA = np.zeros(shape=(9,9))
	for i in range(0,9):
		for j in range(0,9):
			if i==j:
				SIGMA[i][j] = 1.0
			else:
				SIGMA[i][j] = rho

	# print SIGMA
	C = scipy.linalg.cholesky(SIGMA)
	R = np.random.randn(9,numIter)
	Z = (C.transpose()).dot(R)

	total=0.0
	for i in range(0, numIter):
		XT=1.0
		for j in range(0,9):
			XT*=calculateXT(X0[j], sigma[j], T, Z[j][i])
		if XT>=K:
			total+=XT-K
	total/=numIter
	return total

def question3(tc, line):
	numIter = 1000000
	# line = [0.01,0.1,1.,3.,4.5,0.2,0.5,4.5,0.5,1.0,0.7,0.5,0.2,0.01,0.2,0.2,0.001,0.2,0.1,0.2,0.001,0.01]
	rho = float(line[0])
	T = float(line[1])
	# optionType = int(line[2])

	X0 = []
	for i in range(0,9):
		X0.append(float(line[3+i]))

	K = float(line[12])

	sigma = []
	for i in range(0,9):
		sigma.append(float(line[13+i]))

	total = getTotal(rho, numIter, X0, sigma, T, K)
	print str(tc)+",3"+","+'{0:.4f}'.format(total)#str(total)	

def question4(tc, line):
	numIter = 1000
	# line = [0.1,1.,1.0633,3.,4.5,0.2,0.5,4.5,0.5,1.0,0.7,0.5,0.2,0.01,0.2,0.2,0.001,0.2,0.1,0.2,0.001,0.01]
	T = float(line[0])
	# optionType = int(line[1])
	price = float(line[2])

	X0 = []
	for i in range(0,9):
		X0.append(float(line[3+i]))

	K = float(line[12])

	sigma = []
	for i in range(0,9):
		sigma.append(float(line[13+i]))

	minDiff=1000000
	minRho=None
	for rho in np.arange(0.001,1.0,0.001):
		total = getTotal(rho, numIter, X0, sigma, T, K)
		diff = abs(total-price)
		if diff<minDiff:#Find rho that gives minimum error from price
			minDiff=diff
			minRho = rho

	print str(tc)+",4"+","+'{0:.4f}'.format(minRho)#str(minRho)

def getPayoff(rho, numIter, X10, X20, sigma1, sigma2, T, K1, K2, N):
	SIGMA = np.array([[1.0, rho],
					 [rho, 1.0]])
	C = scipy.linalg.cholesky(SIGMA)
	R = np.random.randn(2,numIter)
	Z = (C.transpose()).dot(R)
	
	total=0.0
	for i in range(0, numIter):
		X1T = calculateXT(X10, sigma1, T, Z[0][i])
		X2T = calculateXT(X20, sigma2, T, Z[1][i])

		if X1T<K1 or X2T<K2:
			continue

		payoff = N*(X2T - K2)
		total+=payoff
	total/=numIter

	# print total
	total = int(round(total))
	return total

def question5(tc, line):
	numIter = 1000000
	# line = [0.3,0.3,0.4,1,6,66,15.4,60,10000000]
	# line = [0.3,0.3,0.4,1,6,66,9.4,60,10000000]
	sigma1 = float(line[0])
	sigma2 = float(line[1])
	rho = float(line[2])
	T = float(line[3])
	X10 = float(line[4])
	X20 = float(line[5])
	K1 = float(line[6])
	K2 = float(line[7])
	N = float(line[8])

	payoff = getPayoff(rho, numIter, X10, X20, sigma1, sigma2, T, K1, K2, N)
	print str(tc)+",5"+","+str(payoff)

	# numIter = 1000
	# ppp = []
	# xxx = []
	# for i in np.arange(0.001,1.0,0.001):#sigma1
	# 	payoff = getPayoff(rho, numIter, X10, X20, i, sigma2, T, K1, K2, N)
	# 	ppp.append(payoff)
	# 	xxx.append(i)

	# ppp = []
	# xxx = []
	# for i in np.arange(0.001,1.0,0.001):#sigma2
	# 	payoff = getPayoff(rho, numIter, X10, X20, sigma1, i, T, K1, K2, N)
	# 	ppp.append(payoff)
	# 	xxx.append(i)

	# ppp = []
	# xxx = []
	# for i in np.arange(0.001,1.0,0.001):#rho
	# 	payoff = getPayoff(i, numIter, X10, X20, sigma1, sigma2, T, K1, K2, N)
	# 	# print payoff
	# 	ppp.append(payoff)
	# 	xxx.append(i)

	# plt.plot(xxx, ppp)
	# plt.title('Payoff vs Sigma_2')
	# plt.xlabel('Sigma_2')
	# plt.ylabel('Payoff')
	# plt.show()

if __name__ == "__main__":
	for line in sys.stdin:
		line  = line.strip().split(",")
		tc = int(line[0])
		questionType = int(line[1])

		if questionType==1:
			question1(tc, line[2:])
		elif questionType==2:
			question2(tc, line[2:])		
		elif questionType==3:
			question3(tc, line[2:])		
		elif questionType==4:
			question4(tc, line[2:])		
		else:
			question5(tc, line[2:])
