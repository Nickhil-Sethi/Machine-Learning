'''

Logistic regression trained to recognize handwritten digits between 0-9 (MNIST dataset).
Can achieve ~9% error rate on test set.

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

def sgd_optimization(minibatch_size=600, n_epochs=20, learning_rate=.13, validation_frequency=50):

	# learning rate
	l0 = learning_rate

	# number of examples currently processes 
	num_examples = 0
	
	# number of minibatches to process
	num_minibatches = int(50000/minibatch_size)
	
	# dimensions for classifier
	n_in = numpy.shape(train_set_images[0])[0]
	n_out = 10

	# initializing optimal cost and best validation error
	best_validation_error = numpy.inf 

	# tensor flow session
	sess = tf.Session()

	# classifier imported from logistic regression class
	inp = tf.placeholder(tf.float32, shape=[None,n_in])
	clf = NN.logistic_regression(inp, n_in , n_out,.01)

	# label
	y = tf.placeholder("float",shape=[None,n_out])

	# cost and errors
	cost = clf.cost(y)
	errors = clf.errors(y)

	# gradients
	[gW , gb] = tf.gradients(cost,[clf.W, clf.b])
	
	# update step
	update_W=clf.W.assign_add(-learning_rate*gW)
	update_b=clf.b.assign_add(-learning_rate*gb)

	sess.run(tf.initialize_all_variables())
	e_pr = sess.run(errors, feed_dict={clf.x : test_set_images , y : test_set_labels})

	print "initial error {}%".format(100*float(e_pr)/float(num_test))
	epochs = 1
	counter = 1

	print "training model..."
	while epochs <= n_epochs:
		# iterate through minibatches
		for minibatch_index in xrange(num_minibatches):
			
			# prepping minibatch
			batch_indices = numpy.random.choice(num_train, minibatch_size, replace=False)

			# creating batches
			batch_inputs = [train_set_images[ batch_indices[i] ] for i in xrange(minibatch_size)]
			batch_labels = [train_set_labels[ batch_indices[i] ] for i in xrange(minibatch_size)]

			# run the graph; returns weights,bias,cost,errors
			W,b = sess.run([update_W,update_b],feed_dict={clf.x: batch_inputs , y : batch_labels})

			if counter%validation_frequency==0:

				batch_indices = numpy.random.choice(num_valid, minibatch_size, replace=False)

				batch_inputs  = [valid_set_images[ batch_indices[i] ] for i in xrange(minibatch_size)]
				batch_labels  = [valid_set_labels[ batch_indices[i] ] for i in xrange(minibatch_size)]
				
				e_valid = sess.run(errors, feed_dict={clf.x : batch_inputs , y : batch_labels})

				print "epoch {} minibatch {} validation error {}%".format(epochs , minibatch_index , 100*float(e_valid)/float(minibatch_size))
			counter += 1


		epochs += 1 	

	e_test = sess.run(errors, feed_dict={clf.x : test_set_images , y : test_set_labels})
	print "test error {}%".format(100*float(e_test)/float(num_test))

	return clf

if __name__=='__main__':

	print "...loading data"
	with open('/Users/Nickhil_Sethi/Documents/Datasets/mnist.pkl', 'rb') as f:
	    train_set, valid_set, test_set = pickle.load(f)

	train_set_images 	= train_set[0][:]
	valid_set_images 	= valid_set[0][:]
	test_set_images 	= test_set[0][:]

	num_train			= numpy.shape(train_set_images)[0]
	num_test			= numpy.shape(test_set_images)[0]
	num_valid			= numpy.shape(valid_set_images)[0]

	print "...cleaning data"

	indexer 			= {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9}
	train_set_labels 	=[ msc.one_hot( train_set[1][i] , indexer ) for i in xrange(num_train) ] 
	valid_set_labels 	=[ msc.one_hot( valid_set[1][i] , indexer ) for i in xrange(num_valid) ] 
	test_set_labels 	= [ msc.one_hot( test_set[1][i] , indexer ) for i in xrange(num_test) ] 

	print "data cleaned."
	sgd_optimization(minibatch_size=1000,n_epochs=500)