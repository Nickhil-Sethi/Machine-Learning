import pandas as pd 
import numpy as np
import tensorflow as tf 
import sys
sys.path.insert(0,'/Users/Nickhil_Sethi/Code/Machine-Learning/tensorflow')
import tf_neural_network as NN


cpi 			= pd.read_csv('/Users/Nickhil_Sethi/Documents/Datasets/CPIAUCSL.csv')
gdp 			= pd.read_csv('/Users/Nickhil_Sethi/Documents/Datasets/GDP.csv') 

data 			= pd.merge(cpi,gdp,on='DATE')
data.columns 	= ['DATE','CPI','GDP']

print(data)

# n_data= data.size

# # tensor flow session
# sess = tf.Session()
# n_in = 1
# n_out = 1
# learning_rate = .01

# n_epochs = 100

# # classifier imported from logistic regression class
# inp = tf.placeholder(tf.float32, shape=[None,n_in])
# clf = NN.linear_regression(inp, 1, 1, n_out = 1, init_noise=.5)

# # label
# y = tf.placeholder("float",shape=[None,n_out])

# # cost and error
# cost = clf.cost(y)

# # try including this later; will be necessary for further experiments
# # output = clf.output()

# # gradients
# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# sess.run(tf.initialize_all_variables())

# epochs = 0
# while epochs < n_epochs:
# 	for t in xrange(data.size - 1):

# 		a = [data.loc[t]['CPI']]
# 		p = data.loc[t+1]['CPI']
# 		sess.run(train_step, feed_dict = {inp : a, y : p})

# 	epochs += 1
# 	print "epoch {}".format(epochs)
