import tensorflow as tf
import numpy

class Hidden_Layer(object):


	'''''''''''
	Logistic_Regression class;

	p(y|x) = C * sigm( W * x + b )
	n_out x minibatch_size = C * sigm( n_out x n_in * n_in x minibatch_size + n x 1 )

	'''''''''''

	def __init__(self, v, n_in, n_out):
		
		if not isinstance(v,float):
			raise TypeError('initialization variance must be type float')
		
		self.x = tf.placeholder(tf.float32, shape=[None,n_in])
		self.W = tf.Variable(tf.random_normal([n_in,n_out],stddev=v), name='W')
		self.b = tf.Variable(tf.random_normal([1,n_out],stddev=v), name='b')
	
	def output(self):
		return tf.nn.softmax( tf.matmul(self.x, self.W) + self.b )

	def set_parameters(self,W_new,b_new):

		# set parameters to new values 
		# need to have this as a method, because computation graph uses functions

		self.W.assign(W_new)
		self.b.assign(b_new)

		return self.W,self.b

	def params(self):
		return self.W,self.b
	