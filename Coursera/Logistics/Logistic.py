import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.special import expit
from scipy import optimize


datafile = 'data/ex2data1.txt'     
cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True)
X = np.transpose(np.array(cols[:-1]))
y = np.transpose(np.array(cols[-1:]))
m = y.size
X = np.insert(X,0,1,axis=1)

pos = np.array([X[i] for i in xrange(X.shape[0]) if y[i] == 1])
neg = np.array([X[i] for i in xrange(X.shape[0]) if y[i] == 0])


plt.figure(figsize=(10,6))
plt.plot(pos[:,1],pos[:,2],'k+',label='Admitted')
plt.plot(neg[:,1],neg[:,2],'yo',label='Not admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.grid(True)
#plt.show()

####expit -> sigmoid function###

def h(mytheta, myX):
	return expit(np.dot(myX, mytheta))


def computeCost(mytheta, myX, myy, alpha = 0):
	term1 = np.dot(-np.array(myy).T, np.log(h(mytheta, myX)))
	term2 = np.dot((1 - np.array(myy)).T, np.log(1-h(mytheta, myX)))
	reg = (alpha/2) * np.sum(np.dot(mytheta[1:].T,mytheta[1:]))

	return float((1./m)* np.sum(term1 - term2 + reg))


initial_theta = np.zeros((X.shape[1],1))
#print computeCost(initial_theta,X,y)


def optimizeTheta(mytheta, myX, myy, alpha = 0):
	result = optimize.fmin(computeCost, x0=mytheta, args=(myX, myy, alpha), maxiter=400, full_output=True)
	return result[0], result[1]

theta, mincost = optimizeTheta(initial_theta,X,y)

boundary_xs = np.array([np.min(X[:,1]), np.max(X[:,1])])
boundary_ys = (-1./theta[2])*(theta[0] + theta[1]*boundary_xs)

plt.plot(boundary_xs,boundary_ys,'b-',label='Decision Boundary')
plt.legend()
plt.show()

def makePrediction(mytheta, myx):
    return h(mytheta,myx) >= 0.5


