from __future__ import division

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier

"""Experiment with online SVM vs LR on cancer data"""

class Dataset(object):
	def __init__(self,fname,header=None,index_col=0):
		if '.csv' in fname:
			self.data = pd.read_csv(fname,header=header,index_col=index_col)
		elif '.xl' in fname:
			self.data = pd.read_excel(fname,header=header,index_col=index_col)
	
	def divide(self,frac_train,frac_test):
		NUM_TRAIN				= np.floor(len(self.data)*frac_train)
		NUM_TEST				= len(self.data) - NUM_TRAIN
		self.train_inputs		= self.features[:NUM_TRAIN]
		self.train_labels		= self.labels[:NUM_TRAIN]
		self.test_inputs		= self.features[NUM_TRAIN:]
		self.test_labels 		= self.labels[NUM_TRAIN:]

	def as_matrix(self):
		return self.data.as_matrix()

	def save(self,to_file):
		self.data.to_csv(to_file)

	def __len__(self):
		return len(self.data)

	def shuffle(self):
		self.data = self.data.sample(frac=1.)
	
	def minibatches(self,batch_size):
		if batch_size > len(self.data_):
			raise ValueError('batch size must be less than length of dataset')

		current = 0
		while current+batch_size < len(self.train_inputs):
			batch_features = self.train_inputs[current:current+batch_size]
			batch_labels   = self.train_labels[current:current+batch_size]
			yield batch_features, batch_labels
			current += batch_size

class CancerData(Dataset):
	def __init__(self,fname,header,index_col,frac_train,frac_test,frac_valid=0.):
		Dataset.__init__(self,fname,header,index_col)
		self.data = self.data.sample(frac=1.)
		self.clean_data()
		self.divide(frac_train,frac_test)

	def to_int(self,x):
		x[5] = int(x[5])
		return x 
	
	def clean(self,x):
		return -1 if x == 2 else 1

	def clean_data(self):
		self.data 			= self.data[self.data[6] != '?']
		self.data_ 			= self.data.as_matrix()
		self.features 		= np.array(map(self.to_int,np.array([row[:-1] for row in self.data_])))
		self.labels			= np.array(map(self.clean,np.array([row[-1] for row in self.data_])))

		return self.features,self.labels

class OnlineLR(object):
	def __init__(self,dataset,alpha=.05):
		self.dataset = dataset
		self.alpha   = alpha
		self.model   = SGDClassifier(loss='log',alpha=self.alpha)

	def train(self,n_epochs):
		self.dataset.shuffle()
		
		epoch 				= 0
		while epoch < n_epochs:
			for batch_feats, batch_labels in cancer.minibatches(50):
				self.model.partial_fit(batch_feats,batch_labels,classes=[-1,1])
			epoch += 1
		
		m1 					= sum([1 if x == 1 else 0 for x in self.dataset.test_labels])/len(self.dataset.test_labels)
		benchmark 			= max(m1,1-m1)

		return self.model.score(cancer.train_inputs, cancer.train_labels), self.model.score(cancer.test_inputs, cancer.test_labels)

class OnlineSVM(object):
	def __init__(self,dataset,alpha=.05):
		assert isinstance(dataset,Dataset)
		self.alpha   = alpha
		self.dataset = dataset
		self.model   = SGDClassifier(loss='hinge',alpha=self.alpha)

	def train(self,n_epochs):
		self.dataset.shuffle()
		
		epoch 				= 0
		while epoch < n_epochs:
			for batch_feats, batch_labels in cancer.minibatches(50):
				self.model.partial_fit(batch_feats,batch_labels,classes=[-1,1])
			epoch += 1
		
		m1 					= sum([1 if x == 1 else 0 for x in self.dataset.test_labels])/len(self.dataset.test_labels)
		benchmark 			= max(m1,1-m1)

		return self.model.score(cancer.train_inputs, cancer.train_labels), self.model.score(cancer.test_inputs, cancer.test_labels)

if __name__=='__main__':

	cancer 					= CancerData('/Users/Nickhil_Sethi/Documents/Datasets/Breast_Cancer_Wisconsin.csv',header=None,index_col=0,frac_train=.1,frac_test=.9)	
	NUM_TRAIN				= len(cancer)//10
	NUM_TEST				= len(cancer)-NUM_TRAIN

	check_in                = 15
	n_sim					= 1000
	n_epochs                = 50
	score                   = 0
	for sim in xrange(n_sim):
		svm                 = OnlineSVM(cancer,alpha=.1)
		lr                  = OnlineLR(cancer,alpha=.1)
		svm_train, svm_test = svm.train(n_epochs)
		lr_train, lr_test   = lr.train(n_epochs)
		if (svm_test >= lr_test):
			score += 1
		if sim%check_in==0:
			print "{} simulations ran...".format(sim)
	print "svm outperforms logistic regression {} % of the time".format(100.*score/(n_sim))