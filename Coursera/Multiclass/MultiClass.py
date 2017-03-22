import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import scipy.io 
import scipy.misc
import matplotlib.cm as cm
import random
from scipy.special import expit
from scipy import optimize


datafile = 'data/ex3data1.mat'
mat = scipy.io.loadmat( datafile )
X, y = mat['X'], mat['y']
X = np.insert(X,0,1,axis=1)


def getDatumImg(row):
	width, height = 20,20
	square = row[1:].reshape(width, height)
	return square.T

def displayData(indices_to_display = None):
	width, height = 20,20
	nrows, ncols = 10,10
	if not indices_to_display:
		indices_to_display = random.sample(range(X.shape[0]), nrows*ncols)

	big_picture = np.zeros((height*nrows, width*ncols))
	irow, icol = 0,0
	for idx in indices_to_display:
		if icol==ncols:
				irow+=1
				icol = 0
		iimg = getDatumImg(X[idx])
		big_picture[irow*height:irow*height+iimg.shape[0],icol*width:icol*width+iimg.shape[1]] = iimg
		icol += 1
	fig = plt.figure(figsize=(6,6))
	img = scipy.misc.toimage( big_picture )
	plt.imshow(img,cmap = cm.Greys_r)
	plt.show()

#displayData()

def h(mytheta, myX):
	return expit(np.dot(myX, mytheta))

def computeCost(mytheta, myX, myy, mylambda = 0):
	m = myX.shape[0]
	myh = h(mytheta, myX)
	term1 = np.log(myh).dot(-myy.T)
	term2 = np.log(1.0 - myh).dot((1-myy).T)
	left_hand = (term1 - term2)/m
	right_hand = mytheta.T.dot(mytheta)*mylambda/(2*m)
	return left_hand+right_hand

def costGradient(mytheta, myX, myy, mylambda=0):
	m = myX.shape[0]
	beta = h(mytheta,myX)-myy.T
	regterm = mytheta[1:]*(mylambda/m)
	grad = (1./m)*np.dot(myX.T, beta)
	grad[1:] = grad[1:] + regterm
	return grad


def optimizeTheta(mytheta, myX, myy, mylambda = 0):
	result = optimize.fmin_cg(computeCost, fprime=costGradient, x0=mytheta, 
		args=(myX, myy, mylambda), maxiter=50, disp=False, full_output=True)
	return result[0], result[1]


def buildTheta():
	mylambda = 0
	initialtheta = np.zeros((X.shape[1],1)).reshape(-1)
	#print initialtheta
	Theta = np.zeros((10, X.shape[1]))
	#print Theta
	for i in xrange(10):
		iclass = i if i else 10
		logic_Y = np.array([1 if x == iclass else 0 for x in y])
		itheta, imincost = optimizeTheta(initialtheta,X,logic_Y,mylambda)
		Theta[i,:] = itheta
		#print Theta
		return Theta

Theta = buildTheta()

def predictOnevsAll(mytheta, myrow):
	classes = [10] + range(1,10)
	#print classes
	hypots = [0]*len(classes)
	for i in xrange(len(classes)):
		hypots[i] = h(mytheta[i],myrow)
		#print hypots
	return classes[np.argmax(np.array(hypots))]


n_correct, n_total = 0., 0.
incorrect_indices = []
for irow in xrange(X.shape[0]):
    n_total += 1
    if predictOnevsAll(Theta,X[irow]) == y[irow]: 
        n_correct += 1
    else: incorrect_indices.append(irow)
print "Training set accuracy: %0.1f%%"%(100*(n_correct/n_total))


datafile = 'data/ex3weights.mat'
mat = scipy.io.loadmat( datafile )
Theta1, Theta2 = mat['Theta1'], mat['Theta2']

def propagateForward(row,Thetas):
    """
    Function that given a list of Thetas, propagates the
    Row of features forwards, assuming the features already
    include the bias unit in the input layer, and the 
    Thetas need the bias unit added to features between each layer
    """
    features = row
    for i in xrange(len(Thetas)):
        Theta = Thetas[i]
        z = Theta.dot(features)
        a = expit(z)
        if i == len(Thetas)-1:
            return a
        a = np.insert(a,0,1) #Add the bias unit
        features = a

def predictNN(row,Thetas):
    """
    Function that takes a row of features, propagates them through the
    NN, and returns the predicted integer that was hand written
    """
    classes = range(1,10) + [10]
    output = propagateForward(row,Thetas)
    return classes[np.argmax(np.array(output))]

# "You should see that the accuracy is about 97.5%"
myThetas = [ Theta1, Theta2 ]
n_correct, n_total = 0., 0.
incorrect_indices = []
#Loop over all of the rows in X (all of the handwritten images)
#and predict what digit is written. Check if it's correct, and
#compute an efficiency.
for irow in xrange((X.shape[0])):
    n_total += 1
    if predictNN(X[irow],myThetas) == int(y[irow]): 
        n_correct += 1
    else: incorrect_indices.append(irow)
print "Training set accuracy: %0.1f%%"%(100*(n_correct/n_total))

for x in xrange(5):
    i = random.choice(incorrect_indices)
    fig = plt.figure(figsize=(3,3))
    img = scipy.misc.toimage( getDatumImg(X[i]) )
    plt.imshow(img,cmap = cm.Greys_r)
    predicted_val = predictNN(X[i],myThetas)
    predicted_val = 0 if predicted_val == 10 else predicted_val
    fig.suptitle('Predicted: %d'%predicted_val, fontsize=14, fontweight='bold')