import tensorflow as tf
import tf_logistic_regression as LR
import numpy as np

import pandas as pd


'''

loading data

'''


sa_heartdisease = pd.read_csv('/Users/Nickhil_Sethi/Documents/Datasets/South_African_HeartDisease.csv')
print sa_heartdisease.columns

n_in = len(sa_heartdisease.columns)-2
n_out = 2

num_data=len(sa_heartdisease)

num_train=300
num_valid =0
num_test = num_data - (num_train + num_valid)

print num_test

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

print "benchmark: {}".format(100*float(num_chd)/float(num_data))
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



learning_rate=.000014

sess = tf.Session()

# classifier object; variance of initial 
inp = tf.placeholder(tf.float32, shape=[None,n_in])
clf = LR.Logistic_Regression(inp,.01, n_in , n_out)

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

# parameters 
params = clf.params()

# initialize variables
sess.run(tf.initialize_all_variables())

# compute inital error
e1 = sess.run(errors,feed_dict={clf.x: training_inputs , y : training_labels})
print "initial train error {}%".format(100*float(e1)/float(num_train))


##### sgd_optimization #######

batch_size=60
assert batch_size <= num_train

n_epochs=1
epochs=0

num_minibatches = 100
best_err = np.inf
while epochs <= n_epochs:

	for minibatch_index in xrange( num_minibatches ):


		batch_indices = np.random.choice(xrange(num_train), num_train, replace=False)

		batch_inputs = [training_inputs[ batch_indices[i] ] for i in xrange(batch_size)]
		batch_labels = [training_labels[ batch_indices[i] ] for i in xrange(batch_size)]

		sess.run([update_W,update_b],feed_dict={clf.x: batch_inputs , y: batch_labels})

	epochs += 1

#sess.run(clf.set_parameters(w_opt,b_opt))

e2 = sess.run(errors,feed_dict={clf.x: training_inputs , y : training_labels})
print "train error {}%".format(100*float(e2)/float(num_train))

et = sess.run(errors,feed_dict={clf.x: test_inputs , y : test_labels})
print "test error {}%".format(100*float(et)/float(num_test))

