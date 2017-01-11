'''

Library of machine learning objects used to build neural networks

'''

import tensorflow as tf
import numpy as np


class linear_regression(object):

	''''''''''
	Linear Regression object
	
	'''''''''

	def __init__(self,input,minibatch_size,n_in,n_out=1,init_noise=.01):
		self.x = input
		self.init_noise = init_noise
		self.W = tf.Variable(tf.random_normal([n_in,n_out],stddev=self.init_noise), name='W')
		self.b = tf.Variable(tf.random_normal([1,n_out],stddev=self.init_noise), name='b')
		self.y_pred = tf.matmul(self.x, self.W) + self.b 


	def set_parameters(self,W_new,b_new):

		# set parameters to new values 
		# need to have this as a method, because TensorFlow computation graphs use functions only

		self.W.assign(W_new)
		self.b.assign(b_new)

		return self.W,self.b

	def params(self):
		return self.W,self.b

	def prediction(self):
		return self.y_pred

	def cost(self, y_):
		e = tf.sub(y_, self.prediction())
		return tf.matmul( tf.transpose(e), e )



class logistic_regression(object):


	'''''''''''
	Logistic_Regression class;

	p(y|x) = C * sigm( W * x + b )
	n_out x minibatch_size = C * sigm( n_out x n_in * n_in x minibatch_size + n x 1 )

	'''''''''''

	def __init__(self,input,n_in,n_out,v=.01):

		self.x = input 
		self.W = tf.Variable(tf.random_normal([n_in,n_out],stddev=v), name='W')
		self.b = tf.Variable(tf.random_normal([1,n_out],stddev=v), name='b')
		self.p_y_given_x = tf.nn.softmax( tf.matmul(self.x, self.W) + self.b )
		self.y_pred = tf.argmax(self.p_y_given_x, dimension=1)

	def set_parameters(self,W_new,b_new):

		# set parameters to new values 
		# need to have this as a method, because computation graph uses functions

		self.W.assign(W_new)
		self.b.assign(b_new)

		return self.W,self.b

	def params(self):
		return self.W,self.b

	def p(self):
		return self.p_y_given_x

	def negative_log_likelihood(self,y_):
		likelihood = tf.mul(y_, self.p_y_given_x)  
		return -tf.log(tf.reduce_sum( likelihood, 1, keep_dims = True))
		
	def cost(self, y_):
		return tf.reduce_mean(self.negative_log_likelihood(y_)) 

	def errors(self, y_):
		incorrect_prediction = tf.not_equal( tf.argmax(y_,1), self.y_pred )
		accuracy = tf.reduce_sum(tf.cast(incorrect_prediction, tf.float32))
		return accuracy 

	def prediction(self):
		return self.y_pred


class hidden_layer(object):


	'''''''''''
	Hidden_Layer class;

	output(x | W,b) = C * sigm( W * x + b )
	n_out x minibatch_size = C * sigm( n_out x n_in * n_in x minibatch_size + n x 1 )

	'''''''''''

	def __init__(self,input, n_in, n_out, v=.01):
		
		self.x = input
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

'''
class conv_layer(object):

'''

class neural_network(object):
	def __init__(self,input,dimensions,init_noise=.01):

		#layer = []

		self.x = input
		self.dimensions=dimensions
		self.num_layers = len(dimensions)-1
		self.init_noise=init_noise
		self.layer = list()

		for layer_index in range(self.num_layers-1):
			if layer_index == 0:
				self.layer.append(hidden_layer(input=self.x, v=self.init_noise, n_in=self.dimensions[0], n_out=self.dimensions[1]) )
			else:
				self.layer.append(hidden_layer(input=self.layer[layer_index-1].output(), v=self.init_noise, n_in=self.dimensions[layer_index], n_out=self.dimensions[layer_index+1]) )

		self.layer.append(

			logistic_regression(
			input=self.layer[self.num_layers-2].output(),
			n_in=self.dimensions[self.num_layers-1],
			n_out=self.dimensions[self.num_layers],
			v=self.init_noise
			)
			
		)
	def output(self):
		self.layer[self.num_layers-1].p()

	def cost(self, y_):
		return self.layer[self.num_layers-1].cost(y_)

	def errors(self, y_):
		return self.layer[self.num_layers-1].errors(y_)
