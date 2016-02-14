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



learning_rate = .01

train = theano.function(
	inputs = [index],
	outputs = p.y_hat[index],
	updates = [(p.W, p.W - learning_rate*(y[index] - p.y_hat[index])*V)],
	)

print "y_hat from train",train(1)
print "W updated = ",p.W.get_value()
