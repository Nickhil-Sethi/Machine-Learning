# Machine-Learning

My own code implementing different methods from ML/AI/statistics on different datasets.

Everything here is implemented with Google's TensorFlow library; files beginning with tf (e.g. tf_neural_network.py) denote modules I've written to code basic machine learning objects, like logistic_regression(), linear_regression(), hidden_layer(), etc.

MNIST_logistic_regression.py : implementation of logistic regression for multi-class classification on MNIST data set of handwritten digits.

MNIST_neuralnet.py : implementation of neural network for multi-class classification on MNIST data set of handwritten digits. Neural network takes object takes shape as input.

MISC_library.py : libary of miscelaneous functions for ML, e.g. normal data generator, one-hot vector function, etc.

sa_heart_disease_logistic_regression.py : logistic resgression implemented on data set of south african heart disease patients. Task is to predict heart disease (=1) or no heart disease (=0).

sa_heart_disease_neural_network.py : similar, but with neural network.

simulated_data_logistic_regression.py : logistic regression on simulated normal data. predict y \in {0,1}
