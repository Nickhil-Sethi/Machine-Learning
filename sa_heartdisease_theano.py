import theano
import theano.tensor as T
import numpy as np
from copy import copy

import pandas as pd

sa_heartdisease = pd.read_csv('/Users/Nickhil_Sethi/Documents/Datasets/South_African_HeartDisease.csv')
print sa_heartdisease.columns

n_in = len(sa_heartdisease.columns)-2
n_out = 2

num_data=len(sa_heartdisease)
num_train=150
num_valid =150
num_test = num_data - (num_train + num_valid)

print num_test

assert num_train + num_valid + num_test == num_data 
assert num_test > 0

# clean data, once and for all!
inputs=np.zeros( (n_in, num_data) )
label= np.zeros( (n_out,num_data) )

count = 0
for row in sa_heartdisease.itertuples():

	if row[6]=='Present':
		inputs[4][count] = 1
	else:
		inputs[4][count] = 0
	
	inputs[:4][count]=row[2:6]
	inputs[5:][count]=row[7:11]

	if row[11] == 1:
		label[count]=np.array([0,1]).T
	else:
		label[count]=np.array([1,0]).T

	count+=1

def shared_dataset(data_xy, borrow=True):
	""" Function that loads the dataset into shared variables
	The reason we store our dataset in shared variables is to allow
	Theano to copy it into the GPU memory (when code is run on GPU).
	Since copying data into the GPU is slow, copying a minibatch everytime
	is needed (the default behaviour if the data is not in a shared
	variable) would lead to a large decrease in performance.
	"""

	data_x, data_y = data_xy
	shared_x = theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX),borrow=borrow)
	shared_y = theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX),borrow=borrow)
	# When storing data on the GPU it has to be stored as floats
	# therefore we will store the labels as ``floatX`` as well
	# (``shared_y`` does exactly that). But during our computations
	# we need them as ints (we use labels as index, and if they are
	# floats it doesn't make sense) therefore instead of returning
	# ``shared_y`` we will have to cast it to int. This little hack
	# lets ous get around this issue
	return shared_x, T.cast(shared_y, 'int32')

inputs,label = shared_dataset(inputs,label)

print "benchmark: {}".format(float(sum(label))/float(len(label)))
# divide into train, valid, test

training_inputs = inputs[0:num_train]
training_labels = label[0:num_train]

validation_inputs = inputs[num_train:num_train+num_valid]
validation_labels = label[num_train:num_train+num_valid]

test_inputs = inputs[num_valid+num_train:num_data]
test_labels = label[num_valid+num_train:num_data]

def sgd_optimization():

	return W,b