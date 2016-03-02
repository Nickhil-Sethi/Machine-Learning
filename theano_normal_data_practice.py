import numpy as np
import theano 
import theano.tensor as T
from theano_logistic_regression import LogisticRegression as LR
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# testing logistic regression model with simple binary classification
# on simulated data from normal distributions

# covariance matrices 
sigma1 = np.array([[1.0,0.,0.],[0.,1.4,.0],[0.,0.,1.2]])
sigma2 = np.array([[1.2,0.,0.],[0.,.4,.0],[0.,0,.2]])

# means
m1 = np.array([0.,0.,0.])
m2 = np.array([7.,6.42,6.52])

# dimensions of data
n_in = 3
n_out = 2

assert n_in == np.shape(sigma1)[0]

# initializing 

num_valid = 1000
num_train = 5000

p=.5

valid_set_inputs = np.zeros((n_in,num_valid))
valid_set_labels = np.zeros((n_out,num_valid))

train_set_inputs = np.zeros((n_in,num_train))
train_set_labels = np.zeros((n_out,num_train))

# generating data!
print "generating {} training data points...".format(num_train)

for i in xrange(num_train):
	if np.random.rand() < p:
		train_set_inputs[:,i] = np.random.multivariate_normal(m1,sigma1).T
		train_set_labels[:,i] = np.array([1,0]).T 
	else:
		train_set_inputs[:,i] = np.random.multivariate_normal(m2,sigma2).T
		train_set_labels[:,i] = np.array([0,1]).T 

print "generating {} validation data points...".format(num_valid)

for i in xrange(num_valid):
	if np.random.rand() < p:
		valid_set_inputs[:,i] = np.random.multivariate_normal(m1,sigma1).T
		valid_set_labels[:,i] = np.array([1,0]).T 
	else:
		valid_set_inputs[:,i] = np.random.multivariate_normal(m2,sigma2).T
		valid_set_labels[:,i] = np.array([0,1]).T 

print "data generated."

def sgd_optimization(minibatch_size=100,learning_rate=.13):


    x = T.matrix('x') 
    y = T.ivector('y') 

    clf = LogisticRegression(input=x,n_in=3,n_out=2)

    errors = clf.errors(y)
	return W,b