import tensorflow as tf
import numpy

class Linear_Regression(object):

	''''''''''
	Logistic_Regression class;

	p(y|x) = C * sigm( W * x + b )
	n_out x minibatch_size = C * sigm( n_out x n_in * n_in x minibatch_size + n x 1 )

	'''''''''

	def __init__(self,minibatch_size,n_in,n_out):
		self.x = tf.placeholder("float",shape=[n_in,minibatch_size])
		self.W = tf.Variable(tf.random_normal([1,n_in]), name='W')
		self.b = tf.Variable(tf.random_normal([n_out,1]), name='b')
		self.y = tf.matmul(self.W,self.x) + self.b 


	def set_parameters(self,W_new,b_new):

		# set parameters to new values 
		# need to have this as a method, because computation graph uses functions

		self.W.assign(W_new)
		self.b.assign(b_new)

		return self.W,self.b

	def params(self):
		return self.W,self.b

	def cost(self, y_):
		e = tf.subtract(y_, self.y_pred)
		return tf.matmul( tf.transpose(e), e )

	def prediction(self):
		return self.y_pred
