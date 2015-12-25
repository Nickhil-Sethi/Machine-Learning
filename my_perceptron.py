import sys
sys.path.insert(0,'/Library/Python/2.7/site-packages')

import cPickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as Te

rng = numpy.random 

class perceptron(object):

	def __init__(self, input):
		self.input = input
		self.n_in = numpy.shape(input.get_value())[1]
		self.W = theano.shared(numpy.zeros((self.n_in,1)), name = 'W')
		self.b = theano.shared(rng.randn(1),name = 'b')
		self.y_hat = Te.dot(self.input , self.W) + self.b

	def output(self):
		return Te.gt(self.y_hat,0.0)

def shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=True)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=True)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    #print "!", shared_x.shape
    return shared_x, Te.cast(shared_y, 'int32')

def generate_data(N = 10, p = .5, m1 = numpy.array([0,0]),m2 = numpy.array([20.,30.]),

	cov1 = numpy.eye(2,2),cov2 = numpy.eye(2,2)):	
	
	if(numpy.shape(m1) != numpy.shape(m2)):
		raise TypeError('Dimensions Incompatible')
	else:
		dim = numpy.shape(m1)[0]
		data = numpy.zeros((N,dim))

		labels = numpy.zeros(N)
		for i in range(N):
			r = numpy.random.rand()
			if(r > p):
				data[i] = numpy.random.multivariate_normal(m1,cov1)
				labels[i] = -1
			else:
				data[i] = numpy.random.multivariate_normal(m2,cov2)
				labels[i] = 1

		return [data, labels]

D = generate_data(N = 5)
X,y = shared_dataset(D)
'''
def train(X, y, n_epochs = 100, learning_rate = .00000013, minibatch_size = 20 , validation_frequency = 4):

	N = X.shape
	index = Te.iscalar()

	x = Te.vector()
	y = Te.ivector()

	clf = perceptron(x, n_in = 2)
	error_rate = clf.error_rate(y)

	train_model = theano.function(
		inputs = [index],
		outputs = [error_rate],

		updates = [(clf.W, clf.W + learning_rate*(y_hat - y)*x), 
					(clf.b, clf.b + learning_rate*(y_hat - y))],

		givens = {x: X[index*minibatch_size : index*(minibatch_size+1)], 
				  y: y[index*minibatch_size : (index+1)*(minibatch_size)]  
				 }
		)
	epoch = 0
	M = numpy.floor(N/minibatch_size)
	while epoch <= n_epochs:
		for index in range(M):
			result = train_model(index)
			if index%validation_frequency == 0:
				print "new error rate: ", result

		epoch += 1

	return clf.W.get_value() , clf.b.get_value()
'''
learning_rate = .01
x = Te.vector()
index = Te.iscalar('index')
p = perceptron(X)
print "input 1", p.input.get_value()[1],numpy.shape(p.input.get_value()[1])
print "W = ",p.W.get_value(),numpy.shape(p.W.get_value()),"\n","b = ",p.b.get_value()

train = theano.function(
	inputs = [index],
	outputs = p.y_hat[index],
	updates = [
		(p.W, p.W - learning_rate*(y[index] - p.y_hat[index])*x)
	],
	givens = {x:p.input.get_value()[2]}
	)

print "y_hat from train",train(1)
print "W updated = ",p.W.get_value()
