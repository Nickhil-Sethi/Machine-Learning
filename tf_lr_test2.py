import numpy as np
import tensorflow as tf
import tf_logistic_regression as LR

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# testing logistic regression model with simple binary classification
# on simulated data from normal distributions

# covariance matrices 
sigma1 = np.array([[.5,0.],[0.,.5]])
sigma2 = np.array([[.5,0.],[0.,.5]])

# means
m1 = np.array([0.,0.])
m2 = np.array([4.,4.])

# dimensions of data
n_in = 2
n_out = 2

assert n_in == np.shape(sigma1)[0]
assert n_in == np.shape(m1)[0]
assert (np.shape(sigma1) == np.shape(sigma2))

# initializing 

num_valid = 50
num_train = 200

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
	validation_frequency=30):

	assert patience_increase >= 1
	assert learning_rate > 0
	assert threshold <= 1.0

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
	best_validation_error = np.inf

	while epochs <= n_epochs and not done_looping:

		# iterate through minibatches
		for minibatch_index in xrange(num_minibatches):
				
			# prepping minibatch
			t_inputs = np.zeros( (n_in,minibatch_size) )
			t_labels = np.zeros( (n_out,minibatch_size) )
			
			# pick a random minibatch
			random_start = np.random.randint(1,num_train+1)
			for j in xrange( minibatch_size ):

				t_inputs[:,j] = train_set_inputs[:,(j+random_start)%num_train]
				t_labels[:,j] = train_set_labels[:,(j+random_start)%num_train]

			# run the graph; returns weights,bias,cost,errors,predictions
			(W,b,c,e) = sess.run([update_W,update_b,cost,errors],feed_dict={clf.x: t_inputs , y : t_labels})

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
					best_validation_error = validation_errors


			num_examples += minibatch_size
		
			counter += 1
			if patience <= num_examples:
				print "ran out of patience! {} data points processed. best validation error is {}%".format(num_examples,100*float(best_validation_error)/float(minibatch_size))
				print
				done_looping = True
				break

		epochs += 1 
	
	return clf

clf2 = sgd_optimization(minibatch_size=50,n_epochs=3, learning_rate=.13, \
	patience=num_train, validation_frequency=5)

num_test = 100

p=.5

test_set_inputs = np.zeros((n_in,num_test))
test_set_labels = np.zeros((n_out,num_test))

check_in = 12000
counter = 0
print "generating {} test data points...".format(num_test), "\n"
for i in xrange(num_test):
	if np.random.rand() < p:
		test_set_inputs[:,i] = np.random.multivariate_normal(m1,sigma1).T
		test_set_labels[:,i] = np.array([1,0]).T 
	else:
		test_set_inputs[:,i] = np.random.multivariate_normal(m2,sigma2).T
		test_set_labels[:,i] = np.array([0,1]).T 
	counter += 1
	if counter%check_in==0:
		print "{}% of data points generated".format(100*float(counter)/float(num_test))
print "data generated."

# declare a tf session
sess2 = tf.Session()

# constructing computation graph
clf2 = LR.Logistic_Regression(num_test, n_in , n_out)

# prediction ( given clf.x )
y_hat2 = clf2.prediction()

# label
y2 = tf.placeholder("float",shape=[n_out,num_test])

# cost and errors
cost2 = clf2.cost(y2)
errors2 = clf2.errors(y2)

sess2.run(tf.initialize_all_variables())
print sess2.run(clf2.params())
p = sess2.run(errors2, feed_dict={clf2.x : test_set_inputs, y2 : test_set_labels})

print "\n","{}% error on test set".format(100*float(p)/float(num_test))


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

fig = plt.figure('train sample')
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, y1, z1, zdir='z', c='b',marker='o')
ax.scatter(x2, y2, z2, zdir='z', c='r',marker='^')

plt.draw()
plt.show()
plt.close()



x1 = []
x2 = []
y1 = []
y2 = []
z1 = []
z2 = []
# visualization
for i in xrange(num_test):
	if test_set_labels[0,i] == 1:
		a,b,c = test_set_inputs[:,i].T
		x1.append(a)
		y1.append(b)
		z1.append(c)
	else:
		a,b,c = test_set_inputs[:,i].T
		x2.append(a)
		y2.append(b)
		z2.append(c)

fig2 = plt.figure('test sample')
ax2 = fig2.add_subplot(111, projection='3d')
ax2.scatter(x1, y1, z1, zdir='z', c='b',marker='o')
ax2.scatter(x2, y2, z2, zdir='z', c='r',marker='^')

plt.draw()
plt.show()
plt.close()

