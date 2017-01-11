'''

Logistic regression on south african heart disease data set.
Can achieve ~33% error rate on test set.

Written in Google's tensorflow library.

@author - Nickhil-Sethi

'''

from __future__ import division

import sys
sys.path.insert(0,'/Users/Nickhil_Sethi/Code/Machine-Learning/tensorflow')

import numpy as np
import pandas as pd
import tensorflow as tf
import tf_neural_network as nn

from sklearn.linear_model import SGDClassifier

def LogisticRegressionSGD(training_set,validation_set,test_set):
	learning_rate	=.000014
	sess 			= tf.Session()

	# classifier object; variance of initial 
	inp 			= tf.placeholder(tf.float32, shape=[None,n_in])
	clf 			= nn.logistic_regression(inp,n_in,n_out,.01)

	# label
	y 				= tf.placeholder("float",shape=[None,n_out])

	# cost 
	cost 			= clf.cost(y)

	# gradients
	[gW , gb] 		= tf.gradients(cost,[clf.W, clf.b])
			
	# update step
	update_W		=clf.W.assign_add(-learning_rate*gW)
	update_b		=clf.b.assign_add(-learning_rate*gb)

	# errors
	errors 			= clf.errors(y)

	# pred
	p_y 			= clf.p()

	# parameters 
	params 			= clf.params()

	# initialize variables
	sess.run(tf.initialize_all_variables())

	# compute inital error
	e1 = sess.run(errors,feed_dict={clf.x: training_inputs , y : training_labels})
	print "initial train error {}%".format(100*float(e1)/float(num_train))

	##### sgd_optimization #######

	batch_size=60
	assert batch_size <= num_train

	n_epochs=1
	epochs=0

	num_minibatches = 100
	best_err = np.inf
	while epochs <= n_epochs:
		for minibatch_index in xrange( num_minibatches ):

			batch_indices = np.random.choice(xrange(num_train), num_train, replace=False)
			
			batch_inputs  = [training_inputs[ batch_indices[i] ] for i in xrange(batch_size)]
			batch_labels  = [training_labels[ batch_indices[i] ] for i in xrange(batch_size)]

			sess.run([update_W,update_b],feed_dict={clf.x: batch_inputs , y: batch_labels})

		epochs += 1

	print "benchmark error {}".format()

	e2 = sess.run(errors,feed_dict={clf.x: training_inputs , y : training_labels})
	print "train error {}%".format(100*float(e2)/float(num_train))

	et = sess.run(errors,feed_dict={clf.x: test_inputs , y : test_labels})
	print "test error {}%".format(100*float(et)/float(num_test))

if __name__=='__main__':

	sa_heartdisease 	= pd.read_csv('/Users/Nickhil_Sethi/Documents/Datasets/South_African_HeartDisease.csv')
	sa_heartdisease 	= sa_heartdisease.sample(frac=1.)

	num_data			= len(sa_heartdisease)
	num_train 			= 70
	num_valid 			= 0
	num_test  			= num_data - (num_train + num_valid)
	assert num_train + num_valid + num_test == num_data 

	data 				= sa_heartdisease.as_matrix()
	for row in data:
		if row[5] == 'Absent':
			row[5] = 0
		else:
			row[5] = 1

	inputs 				= [row[:-1] for row in data]
	label 				= [row[-1] for row in data]

	training_inputs 	= inputs[0:num_train]
	training_labels 	= label[0:num_train]

	validation_inputs 	= inputs[num_train:num_train+num_valid]
	validation_labels 	= label[num_train:num_train+num_valid]

	test_inputs 		= inputs[num_valid+num_train:num_data]
	test_labels 		= label[num_valid+num_train:num_data]


	print "traning model..."

	model 				= SGDClassifier(loss='modified_huber',n_iter=500,alpha=.1)
	model.fit(training_inputs,training_labels)

	print "benchmark:               {}".format(100-100*sum(test_labels)/len(test_labels))
	print "training set accuracy    {}".format(100*model.score(training_inputs, training_labels))
	print "test set accuracy        {}".format(100*model.score(test_inputs, test_labels))
	
	print model.coef_[0], model.intercept_
	print sa_heartdisease.columns