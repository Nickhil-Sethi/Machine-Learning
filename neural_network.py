import sys
sys.path.insert(0,'/Library/Python/2.7/site-packages')
import tensorflow as tf
import numpy
import pickle

class Logistic_Regression(object):
	def __init__(self,minibatch_size,n_in,n_out):
		self.x = tf.placeholder("float",shape=[minibatch_size,n_in,1])
		self.W = tf.Variable(tf.random_normal([n_out,n_in], stddev = .1), name='W')
		self.b = tf.Variable(tf.random_normal([n_out,1],stddev = .1), name='b')
		self.out = tf.nn.sigmoid( tf.matmul(self.W,self.x) + self.b )
		self.p_y_given_x = self.out/tf.reduce_sum( self.out )
		self.y_pred = tf.argmax(self.p_y_given_x,dimension = 0)

	def negative_log_likelihood(self,y_):
		likelihood = tf.mul(y_, self.p_y_given_x)  
		return -tf.log( tf.reduce_sum( likelihood, 0, keep_dims = True) )

	def cost(self, y_):
		likelihood = tf.mul(y_, self.p_y_given_x)  
		return tf.reduce_mean( -tf.log( tf.reduce_sum( likelihood, 0, keep_dims = True) ) ) 

	def errors(self, y_):
		correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(self.p_y_given_x,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		return accuracy 

with open('/Users/Nickhil_Sethi/Documents/Datasets/mnist.pkl', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)

train_set_images = train_set[:][0]
train_set_labels = train_set[:][1]


def convert_to_vector(k):
	assert isinstance(k, int)
	assert 0 <= k <= 9
	vec = numpy.zeros(10)
	vec[k] = 1
	return numpy.array([vec]).T

def clean_data(j):
	if not isinstance(j,int):
		raise TypeError('j must be integer')
	d = numpy.array([train_set_images[j]]).T
	v = convert_to_vector(train_set_labels[j])
	return d,v

def sgd_optimization(minibatch_size, validation_frequency, n_epochs=10, learning_rate=.013):

	num_minibatches = int(50000/minibatch_size)
	n_in = numpy.shape(train_set_images[0])[0]
	n_out = 10

	clf = Logistic_Regression(minibatch_size,n_in , n_out)
	y_ = tf.placeholder("float",shape =[minibatch_size,n_out,1])

	cost = clf.cost(y_)
	errors = clf.errors(y_)

	[gW , gb] = tf.gradients(cost,[clf.W, clf.b])
	
	update_W=clf.W.assign_add(learning_rate*gW)
	update_b=clf.b.assign_add(learning_rate*gb)

	sess = tf.Session()
	sess.run(tf.initialize_all_variables())

	epochs = 0
	while epochs <= n_epochs:
		#iterate through minibatches
		for minibatch_index in xrange(num_minibatches):

			#prepping minibatch
			minibatch_images = numpy.zeros( (minibatch_size,n_in,1) )
			minibatch_labels = numpy.zeros( (minibatch_size,n_out,1) )

			#clean data
			for j in xrange( minibatch_index*minibatch_size , (minibatch_index+1)*minibatch_size + 1):
				k = numpy.random.randint(1,50001)
				j_mod_minibatch_size = k%minibatch_size
				minibatch_images[j_mod_minibatch_size,:,:], minibatch_labels[j_mod_minibatch_size,:,:]= clean_data(k%50000)  

			#update weights
			[W,b,c,e] = sess.run([update_W,update_b,cost,errors],feed_dict={clf.x: minibatch_images , y_ : minibatch_labels})

			if epochs == 0 and minibatch_index == 0:
				best_error = e
				best_cost = c
				best_W = W
				best_b = b

			if c < best_cost:
				best_cost = c
				best_error = e
				best_W = W
				best_b = b

			if minibatch_index%validation_frequency==0:

				print "{} cost at minibatch {}".format(best_cost,minibatch_index)
				print "{}% error at minibatch {}".format(100*float(best_error)/float(minibatch_size),minibatch_index)
				print

		epochs += 1 

	return [best_cost, best_error, best_W, best_b]

val = sgd_optimization(65,100,n_epochs=100)

print val[0]
print val[1]
