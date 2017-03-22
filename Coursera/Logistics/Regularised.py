import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.special import expit
from scipy import optimize


datafile = 'data/ex2data2.txt'     
cols = np.loadtxt(datafile,delimiter=',',usecols=(0,1,2),unpack=True)
X = np.transpose(np.array(cols[:-1]))
y = np.transpose(np.array(cols[-1:]))
m = y.size
X = np.insert(X,0,1,axis=1)

pos = np.array([X[i] for i in xrange(X.shape[0]) if y[i] == 1])
neg = np.array([X[i] for i in xrange(X.shape[0]) if y[i] == 0])


def plotData():
	plt.figure(figsize=(6,6))
	plt.plot(pos[:,1],pos[:,2],'k+',label='y=1')
	plt.plot(neg[:,1],neg[:,2],'yo',label='y=0')
	plt.xlabel('Microchip 1')
	plt.ylabel('Microchip 2')
	plt.legend()
	plt.grid(True)
	plt.show()

####expit -> sigmoid function###


def mapFeature(x1col, x2col):
	degrees = 6
	out = np.ones( (x1col.shape[0], 1) )

	for i in range(1, degrees+1):
		for j in range(0, i+1):
			term1 = x1col ** (i-j)
			term2 = x2col ** (j)
			term  = (term1 * term2).reshape( term1.shape[0], 1 ) 
			out   = np.hstack(( out, term ))
	return out

mappedX = mapFeature(X[:,1],X[:,2])

def h(mytheta, myX):
	return expit(np.dot(myX, mytheta))


def computeCost(mytheta, myX, myy, alpha = 0):
	term1 = np.dot(-np.array(myy).T, np.log(h(mytheta, myX)))
	term2 = np.dot((1 - np.array(myy)).T, np.log(1-h(mytheta, myX)))
	reg = (alpha/2) * np.sum(np.dot(mytheta[1:].T,mytheta[1:])) 
	return float((1./m)* np.sum(term1 - term2 + reg))

initial_theta = np.zeros((mappedX.shape[1],1))
print computeCost(initial_theta,mappedX,y)



def optimizeTheta(mytheta, myX, myy, alpha = 0):
	result = optimize.fmin(computeCost, x0=mytheta, args=(myX, myy, alpha), maxiter=400, full_output=True)
	#print result
	return result[0], result[1]

theta, mincost = optimizeTheta(initial_theta,mappedX,y)


def plotBoundary(mytheta, myX, myy, mylambda=0.):
    """
    Function to plot the decision boundary for arbitrary theta, X, y, lambda value
    Inside of this function is feature mapping, and the minimization routine.
    It works by making a grid of x1 ("xvals") and x2 ("yvals") points,
    And for each, computing whether the hypothesis classifies that point as
    True or False. Then, a contour is drawn with a built-in pyplot function.
    """
    theta, mincost = optimizeTheta(mytheta,myX,myy,mylambda)
    xvals = np.linspace(-1,1.5,50)
    yvals = np.linspace(-1,1.5,50)
    zvals = np.zeros((len(xvals),len(yvals)))
    for i in xrange(len(xvals)):
        for j in xrange(len(yvals)):
            myfeaturesij = mapFeature(np.array([xvals[i]]),np.array([yvals[j]]))
            zvals[i][j] = np.dot(theta,myfeaturesij.T)
    zvals = zvals.transpose()

    u, v = np.meshgrid( xvals, yvals )
    mycontour = plt.contour( xvals, yvals, zvals, [0])
    #Kind of a hacky way to display a text on top of the decision boundary
    myfmt = { 0:'Lambda = %d'%mylambda}
    plt.clabel(mycontour, inline=1, fontsize=15, fmt=myfmt)
    plt.title("Decision Boundary")


def makePrediction(mytheta, myx):
	return h(mytheta,myx) >= 0.5

plt.figure(figsize=(12,10))
plt.subplot(221)
plotData()
plotBoundary(theta,mappedX,y,0.)

plt.subplot(222)
plotData()
plotBoundary(theta,mappedX,y,1.)

plt.subplot(223)
plotData()
plotBoundary(theta,mappedX,y,10.)

plt.subplot(224)
plotData()
plotBoundary(theta,mappedX,y,100.)
