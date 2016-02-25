import sys
sys.path.insert(0,'/Library/Python/2.7/site-packages')
import numpy as np
import tensorflow as tf
import tf_logistic_regression as LR

import pickle

# testing logistic regression model with simple binary classification
# on simulated data from normal distributions

# covariance matrices 
sigma1 = np.array([[1.0,0.,0.],[0.,1.4,.0],[0.,0.,1.2]])
sigma2 = np.array([[1.2,0.,0.],[0.,2.4,.0],[0.,0,2.2]])

# means
m1 = np.array([0.,0.,0.])
m2 = np.array([10.,4.42,4.52])

# dimensions of data
n_in = 3
n_out = 2

# initializing 

num_valid = 100000
num_train = 500000

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

def sgd_optimization(minibatch_size=500,learning_rate=.13, n_epochs=100,
	threshold=.995, patience=25000, patience_increase=2., 
	validation_frequency=30, decay=False):

	assert patience_increase >= 1
	assert learning_rate > 0
	assert threshold <= 1.0
	assert isinstance(decay,bool)

	# declare a tf session
	sess = tf.Session()

	# constructing computation graph
	clf = LR.Logistic_Regression(minibatch_size, n_in , n_out)

	# label
	y = tf.placeholder("float",shape=[n_out,minibatch_size])

	# cost and errors
	cost = clf.cost(y)
	errors = clf.errors(y)

	# gradients
	[gW , gb] = tf.gradients(cost,[clf.W, clf.b])
		
	# update step
	update_W=clf.W.assign_add(-learning_rate*gW)
	update_b=clf.b.assign_add(-learning_rate*gb)

	sess.run(tf.initialize_all_variables())

	l0 = learning_rate
	num_minibatches=num_train//minibatch_size

	num_examples=0
	epochs=0
	counter=0
	done_looping=False
	opt_W, opt_b = clf.params()

	while epochs <= n_epochs and not done_looping:

		# iterate through minibatches
		for minibatch_index in xrange(num_minibatches):
				
			# prepping minibatch
			t_inputs = np.zeros( (n_in,minibatch_size) )
			t_labels = np.zeros( (n_out,minibatch_size) )
			
			# pick a random minibatch
			random_start = np.random.randint(1,num_train+1)
			for j in xrange( random_start , random_start + minibatch_size + 1):

				t_inputs[:,j%minibatch_size] = train_set_inputs[:,j%num_train]
				t_labels[:,j%minibatch_size] = train_set_labels[:,j%num_train]

			# run the graph; returns weights,bias,cost,errors,predictions
			(W,b,c,e) = sess.run([update_W,update_b,cost,errors],feed_dict={clf.x: t_inputs , y : t_labels})

			if epochs==0 and minibatch_index==0:
				best_validation_error = e

			# validate model
			if counter%validation_frequency==0:

				print "epoch {} minibatch {}, starting at {}".format(epochs,minibatch_index,random_start)
				print "current cost = {}; error rate {}%".format(c, 100*float(e)/float(minibatch_size)),"\n"
				
				validation_inputs = np.zeros( (n_in,minibatch_size) )
				validation_labels = np.zeros( (n_out,minibatch_size) )
				
				random_start= np.random.randint(1,num_valid+1)

				for j in xrange(minibatch_index*minibatch_size , (minibatch_index+1)*minibatch_size + 1):
					j_mod_valid_size = (j+random_start)%num_valid
					j_mod_minibatch_size = (j)%minibatch_size
					validation_inputs[:,j_mod_minibatch_size] = valid_set_inputs[:,j_mod_valid_size]
					validation_labels[:,j_mod_minibatch_size] = valid_set_labels[:,j_mod_valid_size]

				validation_errors = sess.run(errors, feed_dict={clf.x : validation_inputs, y : validation_labels})
				validation_score = 100*float(validation_errors)/float(minibatch_size)
				
				if validation_score < best_validation_error:
					print "		new best found! cost = {}, error rate = {}%".format(c,validation_score),"\n"
					if validation_score < threshold*best_validation_error:
						patience = max(patience, num_examples*patience_increase)
						print "		threshold reached. patience is now {}...".format(patience)
					[opt_W,opt_b,opt_c,err] = [W,b,c,e]
					best_validation_error = validation_errors


			num_examples += minibatch_size
		
			counter += 1
			if patience <= num_examples:
				print "ran out of patience! {} data points processed. best validation error is {}%".format(num_examples,100*float(best_validation_error)/float(minibatch_size))
				print
				done_looping = True
				break
		if decay:
			learning_rate = float(l0)/np.sqrt( float( 1 + l0*counter ) )
	
		epochs += 1 
	
	return opt_W,opt_b
'''
x1 = []
x2 = []
y1 = []
y2 = []
z1 = []
z2 = []
# visualization
for i in xrange(num_train):
	if train_set_labels[0,i] == 1:
		a,b,c = train_set_inputs[:,i].T
		x1.append(a)
		y1.append(b)
		z1.append(c)
	else:
		a,b,c = train_set_inputs[:,i].T
		x2.append(a)
		y2.append(b)
		z2.append(c)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs1, ys1, zs1, zdir='z', c='b',marker='o')
ax.scatter(xs2, ys2, zs2, zdir='z', c='r',marker='^')

plt.show()
'''
w1,b1 = sgd_optimization(minibatch_size=5000, patience=num_train, patience_increase = 1.2, decay=True)
w2,b2 = sgd_optimization(minibatch_size=5000, patience=num_train, patience_increase = 1.3, decay=True)
w3,b3 = sgd_optimization(minibatch_size=5000, patience=num_train, patience_increase= 1.4, decay=True)
print w1,b1
print w2,b2
print w3,b3