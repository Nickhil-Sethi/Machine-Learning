import numpy as np
import pandas as pd
import tensorflow as tf
import sys
sys.path.insert(0,'/Users/Nickhil_Sethi/Code/Machine-Learning/tensorflow')
import tf_neural_network as NN


'''

loading data

'''

sa_heartdisease = pd.read_csv('/Users/Nickhil_Sethi/Documents/Datasets/South_African_HeartDisease.csv')

#randomly reshuffle rows
sa_heartdisease = sa_heartdisease.sample(frac=1.)

print sa_heartdisease.columns

n_in = len(sa_heartdisease.columns)-2
n_out = 2

num_data=len(sa_heartdisease)

num_train=300
num_valid =0
num_test = num_data - (num_train + num_valid)

print n_in, n_out

assert num_train + num_valid + num_test == num_data 
assert num_test > 0

'''

cleaning data

'''



# clean data, once and for all!
inputs=np.zeros( (num_data, n_in ) )
label= np.zeros((num_data,2))

rowcount = 0
num_chd=0
for row in sa_heartdisease.itertuples():

	if row[6]=='Present':
		inputs[rowcount][4] = 1
	else:
		inputs[rowcount][4] = 0
	
	inputs[rowcount][:4]=row[2:6]
	inputs[rowcount][5:]=row[7:11]

	if row[11]==1:
		label[rowcount]=[1,0]
		num_chd+=1
	else:
		label[rowcount]=[0,1]

	rowcount+=1

print "benchmark: {} \n".format(100*float(num_chd)/float(num_data))
# divide into train, valid, test

training_inputs = inputs[0:num_train]
training_labels = label[0:num_train]

validation_inputs = inputs[num_train:num_train+num_valid]
validation_labels = label[num_train:num_train+num_valid]

test_inputs = inputs[num_valid+num_train:num_data]
test_labels = label[num_valid+num_train:num_data]



'''''

contructing computation graph


'''''

def sgd_optimization(batch_size=60,n_epochs=10,learning_rate=.0013,validation_frequency=30):


	sess = tf.Session()

	# classifier object; variance of initial 

	dim = np.array([9, 5, 2])

	inp = tf.placeholder(tf.float32, shape=[None,n_in])
	clf = NN.neural_network(inp, dim, .01)

	# label
	y = tf.placeholder("float",shape=[None,n_out])

	# cost 
	cost = clf.cost(y)

	# train step
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
			
	# errors
	errors = clf.errors(y)

	# pred
	#p_y = clf.p()

	# initialize variables
	sess.run(tf.initialize_all_variables())

	# compute inital error
	e1 = sess.run(errors,feed_dict={clf.x: training_inputs , y : training_labels})
	print "initial train error {}% \n".format(100*float(e1)/float(num_train))


	##### sgd_optimization #######

	assert batch_size <= num_train

	epochs=0
	num_minibatches = num_train//batch_size
	best_err = np.inf

	print "training for {} batches of size {} over {} epochs \n".format(num_minibatches,batch_size,n_epochs)
	while epochs <= n_epochs:

		for minibatch_index in xrange( num_minibatches ):

			batch_indices = np.random.choice(xrange(num_train), num_train, replace=False)

			batch_inputs = [training_inputs[ batch_indices[i] ] for i in xrange(batch_size)]
			batch_labels = [training_labels[ batch_indices[i] ] for i in xrange(batch_size)]

			sess.run(train_step, feed_dict={inp: batch_inputs , y : batch_labels})

			if minibatch_index%validation_frequency==0:
				err_v = sess.run(errors,feed_dict={inp : batch_inputs , y : batch_labels})
				print "epoch {} batch {} validation error {}".format(epochs, minibatch_index, 100*float(err_v)/float(batch_size))

		epochs += 1

	e2 = sess.run(errors,feed_dict={inp: training_inputs , y : training_labels})
	print "final train error {}%".format(100*float(e2)/float(num_train))

	et = sess.run(errors,feed_dict={inp: test_inputs , y : test_labels})
	print "test error {}%".format(100*float(et)/float(num_test))

	return clf

sgd_optimization(batch_size=num_train,n_epochs=1000,validation_frequency=9,learning_rate=.00053) 