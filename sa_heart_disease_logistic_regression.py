import tensorflow as tf
import tf_logistic_regression as LR
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
inputs=np.zeros( (num_data, n_in ) )
label= np.zeros(num_data)
rowcount = 0
for row in sa_heartdisease.itertuples():

	if row[6]=='Present':
		inputs[rowcount][4] = 1
	else:
		inputs[rowcount][4] = 0
	
	inputs[rowcount][:4]=row[2:6]
	inputs[rowcount][5:]=row[7:11]

	label[rowcount]=row[11]

	rowcount+=1

print "benchmark: {}".format(float(sum(label))/float(len(label)))
# divide into train, valid, test

training_inputs = inputs[0:num_train]
training_labels = label[0:num_train]

validation_inputs = inputs[num_train:num_train+num_valid]
validation_labels = label[num_train:num_train+num_valid]

test_inputs = inputs[num_valid+num_train:num_data]
test_labels = label[num_valid+num_train:num_data]

def sgd_optimization(minibatch_size=30,n_epochs=10,learning_rate=.13,validation_frequency=5,\
	threshold=.995,patience=10*num_train,patience_increase=2,decay=False):

	sess = tf.Session()

	# classifier object
	clf = LR.Logistic_Regression(minibatch_size,n_in,n_out)

	# response variable (minibatch form)
	y = tf.placeholder("float",shape=[n_out,minibatch_size])

	# cost and error on minibatch set
	cost = clf.cost(y)
	error = clf.errors(y)

	# gradients
	[gW , gb] = tf.gradients(cost,[clf.W,clf.b])

	# updates
	update_W = clf.W.assign_add(-learning_rate*clf.W)
	update_b = clf.b.assign_add(-learning_rate*clf.b)

	# initialize session
	sess.run(tf.initialize_all_variables())

	# number of minibatches
	num_minibatches = num_data // minibatch_size
	best_error = np.inf

	l0 = copy(learning_rate)
	counter=0
	epochs=0
	done_looping=False
	num_examples=0
	# iterate through epochs
	while epochs <= n_epochs and (not done_looping):

		# iterate through minibatches, chosen at random
		for minibatch_index in xrange(num_minibatches):

			minibatch_inputs = np.zeros( (n_in,minibatch_size) )
			minibatch_labels = np.zeros( (n_out,minibatch_size) )

			# choose a random initial position
			random_start = np.random.randint(1,num_train)

			for m in xrange(minibatch_size):
				vec=training_inputs[(m+random_start)%(num_train)]
				minibatch_inputs[:,m] = (vec).T

				if training_labels[(m+random_start)%(num_train)] == 1:
					minibatch_labels[:,m] = np.array([0,1]).T
				else:
					minibatch_labels[:,m] = np.array([1,0]).T

			# adjust weights
			[W,b] = sess.run([update_W, update_b],feed_dict={clf.x: minibatch_inputs , y : minibatch_labels})
			[c,e] = sess.run([cost,error],feed_dict={clf.x:minibatch_inputs,y:minibatch_labels})

			num_examples += minibatch_size

			if minibatch_index%validation_frequency==0:
				validation_minibatch_inputs = np.zeros( (n_in, minibatch_size) )
				validation_minibatch_labels = np.zeros( (n_out, minibatch_size) )

				random_start = np.random.randint(1,num_valid)
				for m in xrange(minibatch_size):
					validation_minibatch_inputs[:,m] = (validation_inputs[(m+random_start)%(num_valid)]).T

					if validation_labels[(m+random_start)%(num_valid)] == 1:
						validation_minibatch_labels[:,m] = np.array([0,1]).T
					else:
						validation_minibatch_labels[:,m] = np.array([1,0]).T
				
				# new cost and new error
				[c,e] = sess.run([cost,error], feed_dict={clf.x : validation_minibatch_inputs, y : validation_minibatch_labels})
				if e < best_error:
					print "		new best found! error = {}".format(100*float(e)/float(minibatch_size))
					if e < threshold*best_error:
						patience = max(patience,num_examples*patience_increase)
						print "		threshold reached. patience now {}".format(patience)
					valid_W = W
					valid_b = b

			if patience <= num_examples:
				print "ran out of patience!"
				done_looping = True
				break

			counter+=1
			if decay:
				learning_rate = float(l0)/np.sqrt( float( 1 + l0*counter ) )

		epochs += 1

	return W,b

W,b= sgd_optimization(minibatch_size=100,n_epochs=500,learning_rate=.013,\
	validation_frequency=5,threshold=.9,patience=100*num_train,patience_increase=1.2,decay=True)
print W
print b


sess2 = tf.Session()

# classifier object
clf2 = LR.Logistic_Regression(num_test,n_in,n_out)
clf2.set_parameters(W,b)

# response variable (minibatch form)
y = tf.placeholder("float",shape=[n_out,num_test])

y_hat = clf2.prediction()

# cost and error on minibatch set
cost = clf2.cost(y)
error = clf2.errors(y)

# initialize session
sess2.run(tf.initialize_all_variables())

test_batch_inputs = np.zeros( (n_in,num_test) )
test_batch_labels = np.zeros( (n_out,num_test) )

# choose a random initial position

for m in xrange(num_test):
	vec=test_inputs[m]
	if test_labels[m] == 1:
		test_batch_labels[:,m] = np.array([0,1]).T
	else:
		test_batch_labels[:,m] = np.array([1,0]).T


[c,e] = sess2.run([cost,error], feed_dict={clf2.x : test_batch_inputs, y : test_batch_labels})
print "test error = {}".format(100*float(e)/float(num_test))