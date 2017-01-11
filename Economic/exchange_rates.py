import os
import numpy as np
import pandas as pd 
import tensorflow as tf

from collections import OrderedDict

def load_data(DIR="/Users/Nickhil_Sethi/Documents/Datasets/Exchange Rates"):
	exchange_rates = OrderedDict()
	for file in os.listdir(DIR):
		data_file = os.path.join(DIR,file)
		if ".csv" in data_file:
			exchange_rates[file.replace(".csv","")] = pd.read_csv(data_file,na_values=".")

	combine 				= lambda x,y : x.merge(y,on="DATE")
	exchange_rates  		= reduce(combine,[value for key,value in exchange_rates.items()])
	
	# whitening data
	for column in exchange_rates.columns:
		if column != "DATE":
			exchange_rates[column] = (exchange_rates[column] - exchange_rates[column].mean())/exchange_rates[column].var()
	
	# reset index
	exchange_rates.set_index('DATE',inplace=True)
	return exchange_rates

# vector autoregression model, trained by gradient descent
class VarModel(object):
	def __init__(self,data,lag=1):
		self.lag 		= lag
		self.dimension 	= len(data.columns.values)
		self.data 		= data.as_matrix()
		self.weights 	= tf.Variable(tf.random_normal([self.dimension,self.dimension]), name='W')
		'''
		self.weights 	= OrderedDict()
		
		for i in xrange(self.lag):
			self.weights[i] = tf.Variable(tf.random_normal([dimension,dimension]), name='W_%d') % i 
		'''
		self.bias 		= tf.Variable(tf.random_normal([self.dimension,1]), name='b')

	def y_hat(self,dat):
		return tf.add(tf.matmul(dat,self.weights),self.bias)

	def cost(self,y,inp):
		return tf.matmul(y - inp,y - inp,transpose_b=True)

def sgd_optimization(data,learning_rate=.001,n_epochs=10,num_minibatches=10,minibatch_size=10):
	
	num_col 				= len(data.columns)
	num_steps				= len(data)
	data 					= data.as_matrix()

	sess 				  	= tf.Session()
	weights 				= tf.Variable(tf.random_normal((num_col,num_col)), name='W')
	bias					= tf.Variable(tf.random_normal((num_col,)), name='b')
	
	y 						= tf.placeholder(tf.float32, shape=[num_col])
	inp 					= tf.placeholder(tf.float32, shape=[num_col])
	y_hat					= tf.matmul(weights,inp)
	cost 					= tf.subtract(y,y_hat)
	train_step 				= tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,var_list=[weights,bias])
	sess.run(tf.initialize_all_variables())

	epoch 					= 0
	print(weights.eval())
	while epoch < n_epochs:
		for index,element in enumerate(model.data):
			if index < len(model.data)-1:
				print(data[index])
				sess.run(train_step,feed_dict={inp : data[index], y :data[index+1]})
		print("epoch %d " % epoch)
		epoch += 1

	# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
if __name__=='__main__':
	
	exchange_rates = load_data()
	sgd_optimization(exchange_rates)
# 