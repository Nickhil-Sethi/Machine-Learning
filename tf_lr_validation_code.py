import sys
import tensorflow as tf
import tf_logistic_regression as LR
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# testing logistic regression model for simple binary classification
# on simulated data from normal distributions

w=np.array([[-0.53056282, -0.45644733, -0.52836019],
 [ 3.32600617, -1.80626237,  0.710724  ]] )
b= np.array([[ 3.25907254],
 [-4.01516247]])

# covariance matrices 
sigma1 = np.array([[1.0,0.,0.],[0.,1.4,.0],[0.,0.,1.2]])
sigma2 = np.array([[1.2,0.,0.],[0.,.4,.0],[0.,0,.2]])

# means
m1 = np.array([0.,0.,0.])
m2 = np.array([7.,6.42,6.52])

# dimensions of data
n_in = 3
n_out = 2

# initializing 
num_test = 200000

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

x1 = []
x2 = []
y1 = []
y2 = []
z1 = []
z2 = []
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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1, y1, z1, zdir='z', c='b',marker='o')
ax.scatter(x2, y2, z2, zdir='z', c='r',marker='^')

plt.draw()
plt.show()