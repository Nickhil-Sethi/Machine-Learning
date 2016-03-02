import tensorflow as tf
import tf_logistic_regression as LR
import numpy
import pickle

print "...loading data"

with open('/Users/Nickhil_Sethi/Documents/Datasets/mnist.pkl', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)

train_set_images = train_set[:][0]
train_set_labels = train_set[:][1]

valid_set_images = valid_set[:][0]
valid_set_labels = valid_set[:][1]

test_set_images = test_set[:][0]
test_set_labels = test_set[:][1]

num_train=numpy.shape(train_set_images)[0]
num_test=numpy.shape(test_set_images)[0]
num_valid=numpy.shape(valid_set_images)[0]


def convert_to_vector(k):
	assert isinstance(k, int)
	assert 0 <= k <= 9
	vec = numpy.zeros(10)
	vec[k] = 1
	return numpy.array(vec).T

def clean_data(j,data_set_images,data_set_labels):
	if not isinstance(j,int):
		raise TypeError('j must be integer')

	d = numpy.array(data_set_images[j]).T
	v = convert_to_vector(data_set_labels[j])

	return d,v

def validate_model(W,b,minibatch_size,val_images,val_labels):

	sess = tf.Session()
	# constants
	n_in = numpy.shape(train_set_images[0])[0]
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
	patience_increase = 1.2 , validation_frequency = 5,  decay=False):

	# learning rate
	l0 = learning_rate

	# number of examples currently processes 
	num_examples = 0
	
	# number of minibatches to process
	num_minibatches = int(50000/minibatch_size)

	print "...building model"
	
	# dimensions for classifier
	n_in = numpy.shape(train_set_images[0])[0]
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
	
	counter = 1
	epochs = 1
	done_looping = False

	print "...training model"
	while epochs <= n_epochs and not done_looping:
		# iterate through minibatches
		for minibatch_index in xrange(num_minibatches):
			
			# prepping minibatch
			minibatch_images = numpy.zeros( (n_in,minibatch_size) )
			minibatch_labels = numpy.zeros( (n_out,minibatch_size) )

			# pick a random minibatch
			random_start = numpy.random.randint(1,50001)
			for j in xrange( minibatch_index*minibatch_size , (minibatch_index+1)*minibatch_size + 1):
				j_mod_minibatch_size = (j + random_start)%minibatch_size
				minibatch_images[:,j_mod_minibatch_size], minibatch_labels[:,j_mod_minibatch_size] = clean_data((j+random_start)%num_train, train_set_images,train_set_labels)  

			# run the graph; returns weights,bias,cost,errors
			(W,b) = sess.run([update_W,update_b],feed_dict={clf.x: minibatch_images , y : minibatch_labels})

			# validate model
			if counter%validation_frequency==0:
				# prepping minibatches for validation step
				validation_images = numpy.zeros( (n_in,minibatch_size) )
				validation_labels = numpy.zeros( (n_out,minibatch_size) )

				# pick a random start for the minibatch
				t = numpy.random.randint(1,10001)
				for j in xrange(minibatch_size):
					validation_images[:,(j)%minibatch_size], validation_labels[:,(j)%minibatch_size] = clean_data((t+j)%num_valid, valid_set_images, valid_set_labels)  
				
				#compute zero-one loss on a validation set
				validation_score = validate_model(W,b,minibatch_size,validation_images,validation_labels)

				# printing stuff
				print "epoch {} minibatch {}, starting at {}".format(epochs,minibatch_index,random_start)
				print "cost= {}; error rate= {}%".format(c,100*float(e)/float(minibatch_size))

				if validation_score < best_validation_error:
					if validation_score < threshold*best_validation_error:
						patience = max(patience, num_examples*patience_increase)
						print "		threshold reached. patience is now {}...".format(patience)
					print "		new best found! cost = {}, error rate = {}%".format(c,100*float(validation_score)/float(minibatch_size)),"\n"
					# adjust learning rate 
					[opt_W,opt_b,opt_c,err] = [W,b,c,e]
					best_validation_error = validation_score
					if decay:
						learning_rate = float(l0)/numpy.sqrt( float( 1 + l0*counter ) )
				else:
					print "validation error= {}".format(100*float(validation_score)/float(minibatch_size)),"\n"

			counter += 1
			num_examples += minibatch_size

			# if patience is exhausted, break all loops
			if patience <= num_examples:
				print "ran out of patience! best_validation_error = {}".format(100*float(best_validation_error)/float(minibatch_size))
				done_looping=True
				break

			
		epochs += 1 	

	return W, b

W,b= sgd_optimization(minibatch_size=600,n_epochs=500,learning_rate=.13,patience=200*5000,\
 patience_increase=2., validation_frequency=5, decay=False)

t_images = numpy.zeros((numpy.shape(test_set_images)[1],num_test))
t_labels = numpy.zeros((10,num_test))


for k in xrange(num_test):
	t_images[:,k], t_labels[:,k] = clean_data(k,test_set_images,test_set_labels)

error_test = validate_model(W,b,num_test,t_images,t_labels)
print "test error = {}%".format(100*error_test/float(num_test))