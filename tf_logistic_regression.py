import tensorflow as tf
import numpy

class Logistic_Regression(object):


	''''''''''
	Logistic_Regression class;

	p(y|x) = C * sigm( W * x + b )
	n_out x minibatch_size = C * sigm( n_out x n_in * n_in x minibatch_size + n x 1 )

	'''''''''

	def __init__(self,minibatch_size,n_in,n_out):
		self.x = tf.placeholder("float",shape=[n_in,minibatch_size])
		self.W = tf.Variable(tf.random_normal([n_out,n_in],stddev=.01), name='W')
		self.b = tf.Variable(tf.random_normal([n_out,1],stddev=.01), name='b')
		self.p_y_given_x = tf.nn.softmax( tf.matmul(self.W,self.x) + self.b )
		self.y_pred = tf.argmax(self.p_y_given_x, dimension=0)

	def set_parameters(self,W_new,b_new):

		# set parameters to new values 
		# need to have this as a method, because computation graph uses functions

		self.W.assign(W_new)
		self.b.assign(b_new)

		return self.W,self.b

	def params(self):
		return self.W,self.b

	def negative_log_likelihood(self,y_):
		likelihood = tf.mul(y_, self.p_y_given_x)  
		return -tf.log( tf.reduce_sum( likelihood, 0, keep_dims = True) )

	def cost(self, y_ ):
		return tf.reduce_mean( self.negative_log_likelihood(y_) ) 

	def errors(self, y_):
		incorrect_prediction = tf.not_equal( tf.argmax(y_,0), self.y_pred )
		accuracy = tf.reduce_sum(tf.cast(incorrect_prediction, tf.float32))
		return accuracy 

	def prediction(self):
		return self.y_pred
