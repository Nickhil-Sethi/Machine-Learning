import tensorflow as tf
import tf_logistic_regression as LR
import numpy
import pickle

print "...loading data"

with open('/Users/Nickhil_Sethi/Documents/Datasets/mnist.pkl', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)

num_train=len(train_set[0])
num_valid=len(valid_set[0])
num_test=len(test_set[0])

def convert_to_vector(k):

	assert isinstance(k, int)
	assert 0 <= k <= 9

	vec = numpy.zeros(10)
	vec[k] = 1
	
	return numpy.array(vec).T

train_set_images = train_set[:][0].T
train_set_labels = numpy.array([convert_to_vector(train_set[:][1][j]) for j in xrange(num_train)]).T

valid_set_images = valid_set[:][0].T
valid_set_labels = numpy.array([convert_to_vector( valid_set[:][1][j] ) for j in xrange(num_valid)]).T

test_set_images = test_set[:][0].T
test_set_labels = numpy.array([convert_to_vector(valid_set[:][1][j]) for j in xrange(num_test)]).T

def validate_model(W,b,minibatch_size,val_images,val_labels):

	sess = tf.Session()
	# constants
	n_in = 784
	n_out = 10

	# constructing computation graph
	validator = LR.Logistic_Regression(minibatch_size, n_in , n_out)

	# change parameters
	change = validator.set_parameters(W,b)

	# label
	y = tf.placeholder("float",shape=[n_out,minibatch_size])

	# cost and errors
	cost = validator.cost(y)
	errors = validator.errors(y)

	# initialize session
	sess.run(tf.initialize_all_variables())

	# compute errors
	errs = sess.run(errors, feed_dict={validator.x : val_images , y : val_labels})
	
	return float(errs)

def sgd_optimization(minibatch_size=600, n_epochs=1000, learning_rate=.0013, threshold=.995, patience=5000,
	patience_increase = 2 , validation_frequency = 5,  decay=False):

	# number of examples currently processes 
	num_examples = 0
	
	# number of minibatches to process
	num_minibatches = int(50000/minibatch_size)

	print "...building model"
	
	# dimensions for classifier
	n_in = 784
	n_out = 10

	# initializing optimal cost and best validation error
	best_validation_error = numpy.inf 

	# tensor flow session
	sess = tf.Session()

	# classifier imported from logistic regression class
	clf = LR.Logistic_Regression(minibatch_size, n_in , n_out)

	# label
	y = tf.placeholder("float",shape=[n_out,minibatch_size])

	# cost and errors
	cost = clf.cost(y)
	errors = clf.errors(y)

	# gradients
	[gW , gb] = tf.gradients(cost,[clf.W, clf.b])
	
	# update step
	update_W=clf.W.assign_add(-learning_rate*gW)
	update_b=clf.b.assign_add(-learning_rate*gb)

	sess.run(tf.initialize_all_variables())
	

	epochs = 0
	done_looping = False

	print "...training model"
	while epochs <= n_epochs and not done_looping:
		epochs = epochs + 1

		for minibatch_index in xrange(num_minibatches):
			
			# pick a random minibatch
			random_start = numpy.random.randint(1,50001-minibatch_size)

			minibatch_images = train_set_images[:,random_start:(random_start+minibatch_size)%num_train]
			minibatch_labels = train_set_labels[:,random_start:(random_start+minibatch_size)%num_train]

			# run the graph; returns weights,bias,cost,errors
			W,b = sess.run([update_W,update_b],feed_dict={clf.x: minibatch_images , y : minibatch_labels})
			num_examples += minibatch_size

			# validate model
			if minibatch_index%validation_frequency==0:
				# pick a random start for the minibatch

				random_start = numpy.random.randint(1,num_valid-minibatch_size+1)
				random_end = (random_start + minibatch_size)%num_valid
				
				assert random_start < random_end

				validation_images = valid_set_images[:, random_start:random_end]
				validation_labels = valid_set_labels[:, random_start:random_end]
				
				#compute zero-one loss on a validation set
				validation_score = validate_model(W,b, minibatch_size, validation_images, validation_labels)

				if validation_score < best_validation_error:
					print "		new best found! error rate = {}%".format(100*float(validation_score)/float(minibatch_size))
					
					if validation_score < threshold*best_validation_error:
						print "		threshold reached. patience is now {}...".format(patience)
						patience = max(patience, num_examples*patience_increase)
					
					best_validation_error = validation_score

				else:
					print "epoch {}, minibatch {}, validation error {}%".format(epochs,minibatch_index,100*float(validation_score)/float(minibatch_size))


			# if patience is exhausted, break all loops
			'''
			if patience <= num_examples:
				print "ran out of patience! best_validation_error = {}".format(100*float(best_validation_error)/float(minibatch_size))
				done_looping=True
				break
				'''

	return W, b

W,b= sgd_optimization(minibatch_size=600,n_epochs=500,learning_rate=.13,patience=2*5000,\
 patience_increase=2, validation_frequency=5, decay=False)
