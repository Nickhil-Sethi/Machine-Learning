import numpy as np
import tensorflow as tf
import tf_neural_network as nn
import misc_library
import matplotlib.pyplot as plt


# testing logistic regression model with simple binary classification
# on simulated data from normal distributions

# covariance matrices 
sigma1 = np.array([[.5,0.],[1.,.5]])
sigma2 = np.array([[.5,0.],[0.,.5]])

# means
m1 = np.array([0.,0.])
m2 = np.array([2.,2.])

# dimensions of data
n_in = 2
n_out = 2

assert n_in == np.shape(sigma1)[0]
assert n_in == np.shape(m1)[0]
assert (np.shape(sigma1) == np.shape(sigma2))

# initializing 

num_valid = 50
num_train = 200
num_test = 200

p=.5

learning_rate=.14
train_set_inputs, train_set_labels = misc_library.normal_binary_data(p,num_train,m1,sigma1,m2,sigma2)
valid_set_inputs, valid_set_labels = misc_library.normal_binary_data(p,num_valid,m1,sigma1,m2,sigma2)
test_set_inputs, test_set_labels = misc_library.normal_binary_data(p,num_test,m1,sigma1,m2,sigma2)


#######################
## computation graph ##
#######################


sess = tf.Session()

# constructing computation graph
input = tf.placeholder(tf.float32, shape=[None,n_in])
clf = nn.logistic_regression(input, n_in , n_out,.1)

# label
y = tf.placeholder("float",shape=[None,n_out])

# cost 
cost = clf.cost(y)

# gradients
[gW , gb] = tf.gradients(cost,[clf.W, clf.b])
		
# update step
update_W=clf.W.assign_add(-learning_rate*gW)
update_b=clf.b.assign_add(-learning_rate*gb)

# errors
errors = clf.errors(y)

# pred
p_y = clf.p()

# initialize variables
sess.run(tf.initialize_all_variables())

# compute inital error
e1 = sess.run(errors,feed_dict={clf.x: train_set_inputs , y : train_set_labels})
print "initial train error {}%".format(100*float(e1)/float(num_train))


##### sgd_optimization #######

batch_size=40
n_epochs=10
epochs=0

num_minibatches = 200

while epochs <= n_epochs:


	for minibatch_index in xrange( num_minibatches ):

		batch_indices = np.random.choice(num_train, batch_size, replace=False)

		batch_inputs = [train_set_inputs[ batch_indices[i]] for i in xrange(batch_size)]
		batch_labels = [train_set_labels[ batch_indices[i]] for i in xrange(batch_size)]
		
		sess.run([update_W,update_b],feed_dict={clf.x: batch_inputs , y: batch_labels})

	epochs += 1

e2 = sess.run(errors,feed_dict={clf.x: train_set_inputs , y : train_set_labels})
print "train error {}%".format(100*float(e2)/float(num_train))

et = sess.run(errors,feed_dict={clf.x: test_set_inputs , y : test_set_labels})
print "test error {}%".format(100*float(et)/float(num_test))


x1 = []
x2 = []
y1 = []
y2 = []

# visualization
for i in xrange(num_train):
	if train_set_labels[i,0] == 1:
		a,b = train_set_inputs[i,:]
		x1.append(a)
		y1.append(b)
	else:
		a,b = train_set_inputs[i,:]
		x2.append(a)
		y2.append(b)

fig = plt.figure('train sample')
plt.scatter(x1,y1,c='r')
plt.scatter(x2,y2,c='b')

plt.show()
plt.close()
