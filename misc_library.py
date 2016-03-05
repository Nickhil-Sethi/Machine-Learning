import numpy as np

def one_hot(k,indexer):
	
	# returns one-hot vector with k a key if dict indexer
	if not isinstance(indexer, dict):
		raise TypeError('indexer must be type dict')

	if not k in indexer.keys():
		raise ValueError('k not in keys of indexer')

	vec = np.zeros( len(indexer.keys()) )
	vec[ indexer[k] ] = 1

	return vec


def normal_binary_data(p,num_train,m1,sigma1,m2,sigma2):
	check_in = num_train//33
	counter = 0

	# assert both matrices are square
	if not (np.shape(sigma1)[0] == np.shape(sigma1)[1] and np.shape(sigma2)[0] == np.shape(sigma2)[1]):
		raise TypeError('covariance matrices must be square')

	if not np.shape(sigma1) == np.shape(sigma2):
		raise TypeError('matrices must be same dimensions')

	n_in = np.shape(sigma1)[0]

	train_set_inputs = np.zeros((num_train,n_in))
	train_set_labels = np.zeros((num_train,n_in))

	for i in xrange(num_train):
		if np.random.rand() < p:
			train_set_inputs[i,:] = np.random.multivariate_normal(m1,sigma1)
			train_set_labels[i,:] = np.array([1,0])
		else:
			train_set_inputs[i,:] = np.random.multivariate_normal(m2,sigma2)
			train_set_labels[i,:] = np.array([0,1])

		counter += 1
		if check_in > 0 and counter%check_in==0:
			print "{}% data points generated".format(100*float(counter)/float(num_train))

	return train_set_inputs, train_set_labels