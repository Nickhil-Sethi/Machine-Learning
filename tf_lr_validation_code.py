import sys
sys.path.insert(0,'/Library/Python/2.7/site-packages')
import tensorflow as tf
import tf_logistic_regression as LR
import numpy as np
import pickle

# testing logistic regression model for simple binary classification
# on simulated data from normal distributions

w=np.array([[-0.09544643, -0.16063242 ,-0.16268137],
 [ 0.06587631 , 0.08666617 , 0.08911248]])
b= np.array([[ 0.90085769],
 [-1.13099718]])

# covariance matrices 
sigma1 = np.array([[1.0,0.,0.],[0.,1.4,.0],[0.,0.,1.2]])
sigma2 = np.array([[1.2,0.,0.],[0.,2.4,.0],[0.,0,2.2]])

# means
m1 = np.array([0.,0.,0.])
m2 = np.array([10.,20.42,20.52])

# dimensions of data
n_in = 3
n_out = 2

# initializing 
num_test = 1000000

p=.5

test_set_inputs = np.zeros((n_in,num_test))
test_set_labels = np.zeros((n_out,num_test))

check_in = 50000
counter = 0
print "generating {} test data points...".format(num_test)
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
sess = tf.InteractiveSession()

# constructing computation graph
clf = LR.Logistic_Regression(num_test, n_in , n_out)
change = clf.set_parameters(w,b)

# prediction ( given clf.x )
y_hat = clf.prediction()

# label
y = tf.placeholder("float",shape=[n_out,num_test])

# cost and errors
cost = clf.cost(y)
errors = clf.errors(y)

sess.run(tf.initialize_all_variables())
sess.run(change)

p = sess.run(errors, feed_dict={clf.x : test_set_inputs, y : test_set_labels})

print "\n","{}% error on test set".format(100*float(p)/float(num_test))