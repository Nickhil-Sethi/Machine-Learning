import sys
sys.path.insert(0,'/Users/Nickhil_Sethi/Code/Machine-Learning/Homework-Answers')

import numpy

from hw1_sgd import hw1_skeleton_code
from scipy.optimize import minimize

def ridge(Lambda):
	def ridge_obj(theta):
		return ((numpy.linalg.norm(numpy.dot(X,theta) - y))**2)/(2*N) + Lambda*(numpy.linalg.norm(theta))**2
	return ridge_obj

def compute_loss(Lambda, theta):
	return ((numpy.linalg.norm(numpy.dot(X,theta) - y))**2)/(2*N)

class LinearSystem(object):
	"""data matrix of dimensions m x d (num_examples x num_dimensions)"""
	def __init__(self,m=150,d=75,p=.5):
		self.m = m
		self.d = d
		self.p = p
		self.design_matrix()
		self.theta()
		self.y()

	def design_matrix(self):
		self.X = numpy.random.rand((self.m,self.d))

	def theta(self):
		self.theta = numpy.zeros(self.d)
		for i in xrange(10):
			if np.random.rand() < self.p:
				self.theta[i] = 10.
			else:
				self.theta[i] = -10.

	def y(self):
		self.y = self.X.dot(self.theta) + np.random.randn(self.m)

	def split(self):
		pass

if False:
	X 			= numpy.loadtxt("X.txt")
	y 			= numpy.loadtxt("y.txt")
	(N,D) 		= X.shape
	w 			= numpy.random.rand(D,1)

	for i in range(-5,6):
		Lambda 	= 10**i
		w_opt 	= minimize(ridge(Lambda), w)
		print Lambda, compute_loss(Lambda, w_opt.x)