import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import scipy.io 
import scipy.misc
import matplotlib.cm as cm
import random
from scipy.special import expit
from scipy import optimize
import  itertools


datafile = 'data/ex4data1.mat'
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



input_layer_size = 400
hidden_layer_size = 25
output_layer_size = 10 
n_training_samples = X.shape[0]



def flattenParams(thetas_list):
    """
    Hand this function a list of theta matrices, and it will flatten it
    into one long (n,1) shaped numpy array
    """
    flattened_list = [ mytheta.flatten() for mytheta in thetas_list ]
    #print flattened_list
    combined = list(itertools.chain.from_iterable(flattened_list))
    #print combined
    assert len(combined) == (input_layer_size+1)*hidden_layer_size + \
                            (hidden_layer_size+1)*output_layer_size
    return np.array(combined).reshape((len(combined),1))

def reshapeParams(flattened_array):
    theta1 = flattened_array[:(input_layer_size+1)*hidden_layer_size] \
            .reshape((hidden_layer_size,input_layer_size+1))
    theta2 = flattened_array[(input_layer_size+1)*hidden_layer_size:] \
            .reshape((output_layer_size,hidden_layer_size+1))
    
    return [ theta1, theta2 ]

def flattenX(myX):
    return np.array(myX.flatten()).reshape((n_training_samples*(input_layer_size+1),1))

def reshapeX(flattenedX):
    return np.array(flattenedX).reshape((n_training_samples,input_layer_size+1))




def computeCost(mythetas_flattened, myX_flattened, myy, mylambda=0):
	mythetas = reshapeParams(mythetas_flattened)
	myX = reshapeX(myX_flattened)
	m = n_training_samples
	total_cost = 0
	for irow in xrange(m):
		myrow = myX[irow]
		myhs = propagateForward(myrow, mythetas)[-1][1]
		tmpy = np.zeros(output_layer_size, 1)
		tmpy[myy[irow] - 1] = 1
		mycost = -tmpy.T.dot(np.log(myhs)) - (1 - tmpy.T).dot(np.log(1-myhs))
		total_cost += mycost

	total_cost = float(total_cost)/m 
	total_reg = 0

	for mytheta in mythetas:
		total_reg += np.sum(mytheta*mytheta)

	total_reg *= float(mylambda)/(2*m)
	return total_cost+total_reg	


def propagateForward(row, thetas):
	features = row
	zs_as_per_layer = []
	for i in xrange(len(thetas)):
		theta = theta[i]
		z = theta.dot(features).reshape((theta.shape[0],1))
		a = expit(z)
		zs_as_per_layer.append((z, a))
		if i == len(thetas)-1:
			return np.array(zs_as_per_layer)
		a = np.insert(a,0,1)
		features = a

def sigmoidGradient(z):
	dummy = expit(z)
	return dummy*(1-dummy)

def getRandThetas():
	epsilon_init = 0.2
	theta1_shape = (hidden_layer_size, input_layer_size+1)
	theta2_shape = (output_layer_size, hidden_layer_size+1)
	rand_thetas = [np.random.rand(*theta1_shape)*2*epsilon_init - epsilon_init, np.random.rand(*theta2_shape)*2*epsilon_init - epsilon_init]
	return rand_thetas

#myThetas = getRandThetas()	

def backPropagate(mythetas_flattened, myX_flattened, myy, mylambda=0):
	mythetas = reshapeParams(mythetas_flattened)
	myX = reshapeX(myX_flattened)
	m = n_training_samples
	Delta1 = np.zeros(hidden_layer_size,input_layer_size+1)
	Delta2 = np.zeros(output_layer_size,hidden_layer_size+1)
	for irow in xrange(m):
		myrow = myX[irow]
		a1 = myrow.reshape((input_layer_size+1,1))
		temp = propagateForward(myrow, mythetas)
		z2 = temp[0][0]
		a2 = temp[0][1]
		z3 = temp[1][0]
		a3 = temp[1][1]
		tmpy = np.zeros(10,1)
		tmpy[myy[irow]-1] = 1
		delta3 = a3 - tmpy
		delta2 = mythetas[1].T[1:,:].dot(delta3)*sigmoidGradient(z2)
		a2 = np.insert(a2,0,1,axis=0)
		Delta1 += delta2.dot(a1.T)
		Delta2 += delta1.dot(a2.T)

	D1 = Delta1/float(m)
	D2 = Delta2/float(m)

	D1[:,1:] = D1[:,1:] + (float(mylambda)/m)*mythetas[0][:,1:]
	D2[:,1:] = D2[:,1:] + (float(mylambda)/m)*mythetas[1][:,1:]	
	return flattenParams([D1,D2].flatten())

#flattenedD1D2 = backPropagate(flattenParams(myThetas), flattenX(X), y, mylambda = 1)
#D1,D2 = reshapeParams(flattenedD1D2)

def checkGradient(mythetas,myDs,myX,myy,mylambda=0.):
    myeps = 0.0001
    flattened = flattenParams(mythetas)
    flattenedDs = flattenParams(myDs)
    myX_flattened = flattenX(myX)
    n_elems = len(flattened) 
    #Pick ten random elements, compute numerical gradient, compare to respective D's
    for i in xrange(10):
        x = int(np.random.rand()*n_elems)
        epsvec = np.zeros((n_elems,1))
        epsvec[x] = myeps
        cost_high = computeCost(flattened + epsvec,myX_flattened,myy,mylambda)
        cost_low  = computeCost(flattened - epsvec,myX_flattened,myy,mylambda)
        mygrad = (cost_high - cost_low) / float(2*myeps)
        print "Element: %d. Numerical Gradient = %f. BackProp Gradient = %f."%(x,mygrad,flattenedDs[x])	


def trainNN(mylambda=0):
	randomThetas_unrolled = flattenParams(getRandThetas())
	result = scipy.optimize.fmin_cg(computeCost, x0=randomThetas_unrolled, fprime=backPropagate, args=(flattenX(X),mylambda), maxiter=100, disp=True, full_output=True)
	return reshapeParams(result[0])


learned_thetas = trainNN()


def predictNN(row, Thetas):
	classes = range(1,10) + [10]
	output = propagateForward(row, Thetas)
	return classes[np.argmax(output[-1][1])]


def computeAccuracy(myX,myThetas,myy):
	n_correct, n_total = 0, myX.shape[0]
	for irow in xrange(n_total):
		if int(predictNN(myX[irow],myThetas)) == int(myy[irow]): 
			n_correct += 1
	print "Training set accuracy: %0.1f%%"%(100*(float(n_correct)/n_total))


computeAccuracy(X,learned_Thetas,y)









