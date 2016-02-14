import numpy
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
rng = numpy.random

N = 1000
Te = 10000
feats = 50
v = 1.1
l = .1

def generate_sample_data(N, Te, feats,v):
    print 'generating data...','\n'
    beta = rng.randn(feats)
    b0 = rng.randn(1)[0]
    d_train = 100*rng.randn(N,feats)
    l_train = d_train.dot(beta) + b0 + rng.normal(0,v,(N,1))[0]
    d_test = 100*rng.randn(Te,feats)
    l_test = d_test.dot(beta) + b0 + rng.normal(0,v,(N,1))[0]

    return [d_train,l_train],[d_test,l_test], [beta,b0]
def shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=True)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=True)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    #print "!", shared_x.shape
    return [shared_x, shared_y]

class regularized_linear_regression(object):
    def __init__(self, input, N, feats, l):
        self.input = input
        self.w = theano.shared(rng.randn(feats), name="w")
        self.b = theano.shared(rng.randn(1)[0], name="b")
        self.capacity = T.dot(self.w , self.w)
        self.y_hat = T.dot(input, self.w) + self.b
        self.l = l

    def cost(self, y):
		#detached from others because 'y' is chosen at runtime
		return T.mean((y - self.y_hat)**2) + self.l*self.capacity


def sgd_optimization_mnist(train_set, test_set, learning_rate=0.00000013, n_epochs=1000, 
	num_minibatches=50, validation_frequency=5):
    """
    Demonstrate stochastic gradient descent optimization of a log-linear
    model

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz

    """
    print 'linear regression with l2 penalty'
    print 'N = %d, Te = %d, feats = %d' %(N,Te,feats)
    train_set_x, train_set_y = test_set
    test_set_x, test_set_y = train_set

    minibatch_size = int( numpy.floor(N/num_minibatches) )

    print 'minibatch size = %d' %(minibatch_size), '\n'


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print 'building the model...'

    # allocate symbolic variables for the data
    index = T.iscalar('index')
    # generate symbolic variables for input (x and y represent a
    # minibatch)
    x = T.matrix('x')  # data, presented as rasterized images
    y = T.vector('y')  # labels, presented as 1D vector of labels

    # construct the linear regression class
    # Each MNIST image has size 28*28
    classifier = regularized_linear_regression(x, N, feats,l)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.cost(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch


    # compute the gradient of cost with respect to theta = (W,b)
    g_w = T.grad(cost=cost, wrt=classifier.w)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # start-snippet-3
    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [(classifier.w, classifier.w - learning_rate * g_w),
               (classifier.b, classifier.b - learning_rate * g_b)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index*minibatch_size : (index+1)*minibatch_size],
            y: train_set_y[index*minibatch_size : (index+1)*minibatch_size]       
            }
    )

    test_model =theano.function(
    	inputs = [],
    	outputs = cost,
    	givens={
            x: test_set_x,
            y: test_set_y       
            } 
        )

    epoch = 0
    minc = 1000
    optw = numpy.zeros(feats)
    optb = 0.
    print 'training the model...', '\n'
    while epoch <= n_epochs:
    	#print 'epoch %d' %(epoch)

    	for minibatch_index in range(num_minibatches):
    		c = train_model(minibatch_index)
    		if (minibatch_index%validation_frequency) == 0 and (c < minc):
    			minc = c
    			optw = classifier.w.get_value()
    			optb = classifier.b.get_value()
    			print "new best: error on epoch %d, minibatch %d:" %(epoch,minibatch_index), c
    			
    		elif c < minc:
    			minc = c
    			optw = classifier.w.get_value()
    			optb = classifier.b.get_value()
    		
    	epoch += 1
    err = test_model()
    return err,minc,optw,optb

D1,D2,B = generate_sample_data(N,Te,feats,v)

test_set = shared_dataset(D1)
train_set = shared_dataset(D2)

e,m,w,b = sgd_optimization_mnist(train_set,test_set)

print '\n',"train error: ", m
print "test error: ", e

