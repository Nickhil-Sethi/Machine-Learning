'''

Module trains fully-connected neural net to recognize handwritten digits between 0-9 (MNIST dataset).
Number of layers and shape can be chosen by user in the 'dim' variable. 

Fully connected neural network can achieve ~5% error rate on test set.

Written in Google's tensorflow library.

@author - Nickhil-Sethi

'''

import sys
sys.path.insert(0,'/Users/Nickhil_Sethi/Code/Machine-Learning/tensorflow_objects')

import numpy
import pickle
import tensorflow as tf
import misc_library as msc
import tf_neural_network as NN


def sgd_optimization(dim=numpy.array([784, 784//50, 10]), minibatch_size=600, n_epochs=20, 
	learning_rate=.0013, validation_frequency=50, decay=False):


	'''some constants'''

	learning_rate0 	= learning_rate

	num_minibatches = num_train//minibatch_size
	
	n_in 			= numpy.shape(train_set_images[0])[0]
	n_out 			= 10
	best_validation_error = numpy.inf 

	sess 			= tf.Session()
	inp 			= tf.placeholder(tf.float32, shape=[None,n_in])
	clf 			= NN.neural_network(inp, dim, .5)
	y 				= tf.placeholder("float",shape=[None,n_out])
	cost 			= clf.cost(y)
	errors 			= clf.errors(y)
	# try including this later; will be necessary for further experiments
	# output = clf.output()
	# gradients
	train_step 	= tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
	sess.run(tf.initialize_all_variables())
	e_pr 		= sess.run(errors, feed_dict={clf.x : test_set_images , y : test_set_labels})

	''' optimizing model '''

	print "initial error {}% \n".format(100*float(e_pr)/float(num_test))
	epochs = 1
	counter = 1


	print "training model... \n"

	# TODO: later add early stopping feature in while loop
	while epochs <= n_epochs:
		for minibatch_index in xrange(num_minibatches):
			batch_indices = numpy.random.choice(num_train, minibatch_size, replace=False)
			batch_inputs  = [train_set_images[ batch_indices[i] ] for i in xrange(minibatch_size)]
			batch_labels  = [train_set_labels[ batch_indices[i] ] for i in xrange(minibatch_size)]

			# run one train step
			sess.run(train_step,feed_dict={inp: batch_inputs , y : batch_labels})

			# validation section
			if counter%validation_frequency==0:

				batch_indices = numpy.random.choice(num_valid, minibatch_size, replace=False)
				batch_inputs  = [ valid_set_images[ batch_indices[i] ] for i in xrange(minibatch_size)]
				batch_labels  = [ valid_set_labels[ batch_indices[i] ] for i in xrange(minibatch_size)]
				e_valid 	  = sess.run(errors, feed_dict={clf.x : batch_inputs , y : batch_labels})

				print "epoch {} minibatch {} validation error {}%".format(epochs , minibatch_index , 100*float(e_valid)/float(minibatch_size))

				if decay:
					learning_rate = .999*learning_rate
			
			counter += 1
		epochs += 1 	

	e_test = sess.run(errors, feed_dict={clf.x : test_set_images , y : test_set_labels})
	print "test error {}%".format(100*float(e_test)/float(num_test))
	return clf

if __name__ == '__main__':

	dataset = '/Users/Nickhil_Sethi/Documents/Datasets/mnist.pkl'
	print "...loading {}".format(dataset)
	
	with open(dataset, 'rb') as f:
	    train_set, valid_set, test_set = pickle.load(f)

	train_set_images 	= train_set[0][:]
	valid_set_images 	= valid_set[0][:]
	test_set_images 	= test_set[0][:]

	num_test			= numpy.shape(test_set_images)[0]
	num_train			= numpy.shape(train_set_images)[0]
	num_valid			= numpy.shape(valid_set_images)[0]

	print "...cleaning data {}"

	# converting numbers to "one-hot" vectors; i.e. 2 is converted to [0,0,1,0,0,0,0,0,0,0]
	# indexer assigns each label an index 
	indexer 			= {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9}
	train_set_labels 	= [ msc.one_hot( train_set[1][i] , indexer ) for i in xrange(num_train) ] 
	valid_set_labels 	= [ msc.one_hot( valid_set[1][i] , indexer ) for i in xrange(num_valid) ] 
	test_set_labels 	= [ msc.one_hot( test_set[1][i]  , indexer ) for i in xrange(num_test) ] 

	print "\n data cleaned. \n"

	sgd_optimization(dim=numpy.array([784, 784//36, 10]),minibatch_size=8200,n_epochs=1000,validation_frequency=3,learning_rate=.0013)
