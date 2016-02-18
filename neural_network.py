import sys
sys.path.insert(0,'/Library/Python/2.7/site-packages')
import tensorflow as tf
import tf_logistic_regression as LR
import numpy
import pickle

with open('/Users/Nickhil_Sethi/Documents/Datasets/mnist.pkl', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)

train_set_images = train_set[:][0]
train_set_labels = train_set[:][1]

test_set_images = test_set[:][0]
test_set_labels = test_set[:][1]

print numpy.shape(train_set_images)
print numpy.shape(test_set_images)

def convert_to_vector(k):
	assert isinstance(k, int)
	assert 0 <= k <= 9
	vec = numpy.zeros(10)
	vec[k] = 1
	return numpy.array(vec).T

def clean_data(j,data_set_images,data_set_labels):
	if not isinstance(j,int):
		raise TypeError('j must be integer')

	d = numpy.array(data_set_images[j]).T
	v = convert_to_vector(data_set_labels[j])

	return d,v

def sgd_optimization(minibatch_size=150, validation_frequency=350, n_epochs=100, 
	learning_rate=.0013, decay=False, patience=1):

	# some constants 
	num_minibatches = int(50000/minibatch_size)
	n_in = numpy.shape(train_set_images[0])[0]
	n_out = 10

	# constructing computation graph

	clf = LR.Logistic_Regression(minibatch_size, n_in , n_out)

	y = tf.placeholder("float",shape=[n_out,minibatch_size])

	# cost and errors
	cost = clf.cost(y)
	errors = clf.errors(y)

	# prediction
	y_hat = clf.prediction()

	[gW , gb] = tf.gradients(cost,[clf.W, clf.b])
	
	update_W=clf.W.assign_add(-learning_rate*gW)
	update_b=clf.b.assign_add(-learning_rate*gb)

	sess = tf.Session()
	sess.run(tf.initialize_all_variables())

	counter = 1
	opt_c = 100 
	# optimizing model
	epochs = 1
	while epochs <= n_epochs:
		#iterate through minibatches
		for minibatch_index in xrange(num_minibatches):
			
			#prepping minibatch
			minibatch_images = numpy.zeros( (n_in,minibatch_size) )
			minibatch_labels = numpy.zeros( (n_out,minibatch_size) )

			#pick a random minibatch
			random_start = numpy.random.randint(1,50001)
			for j in xrange( minibatch_index*minibatch_size , (minibatch_index+1)*minibatch_size + 1):
				j_mod_minibatch_size = (j + random_start)%minibatch_size
				minibatch_images[:,j_mod_minibatch_size], minibatch_labels[:,j_mod_minibatch_size] = clean_data((j+random_start)%50000, train_set_images,train_set_labels)  

			#update weights
			(W,b,c,e,yh) = sess.run([update_W,update_b,cost,errors,y_hat],feed_dict={clf.x: minibatch_images , y : minibatch_labels})

			if epochs==1 and minibatch_index==0:
				opt_W = W
				opt_b = b
				err = e
			
			if c < opt_c:
				print "		new best found! cost = {}, error rate = {}%".format(c,100*float(e)/float(minibatch_size)),"\n"
				opt_W = W
				opt_b = b
				opt_c = c
				err = e

			if minibatch_index%validation_frequency==0:
				print "epoch {} minibatch {}, starting at {}".format(epochs,minibatch_index,random_start)
				print "current cost = {}; error rate {}%".format(c, 100*float(e)/float(minibatch_size))
				print

			counter += 1

			if decay and counter > patience:
				learning_rate = learning_rate*numpy.exp(-counter)
			
		epochs += 1 	

	return opt_W, opt_b

W_new,b_new = sgd_optimization(minibatch_size=1000,validation_frequency=300,n_epochs=500,learning_rate=5.13,decay=True,patience=10000)

num_test = numpy.shape(test_set_images)[0]
n_in = numpy.shape(train_set_images[0])[0]
n_out = 10

clean_test_set_images = numpy.zeros((784,num_test))
clean_test_set_labels = numpy.zeros((10,num_test))

clf2=LR.Logistic_Regression(num_test, n_in, n_out)
change_params=clf2.set_parameters(W_new,b_new)
Wn,bn=clf2.params()
# constructing computation graph
y2 = tf.placeholder("float",shape=[n_out,num_test])

# cost and errors
errors2 = clf2.errors(y2)
sess = tf.Session()
sess.run(tf.initialize_all_variables())

for j in xrange(num_test):
	clean_test_set_images[:,j], clean_test_set_labels[:,j] = clean_data(j,test_set_images,test_set_labels)

test_err,p1,p2 = sess.run([errors2,Wn,bn], feed_dict={clf2.x : clean_test_set_images , y2 : clean_test_set_labels})
percent_error = 100*float(test_err)/float(num_test)	

assert p1.all()==W_new.all()

print "{}% error on test set".format(percent_error)