"""Experiments with Latent Dirichlet Allocation for compressing and detecting spam texts"""

from time import time

import lda
import numpy as np
import scipy as sp
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer

def check_probabilities(topic_word):
	for n in range(5):
	    sum_pr 			= sum(topic_word[n,:])
	    print("topic: {} sum: {}".format(n, sum_pr))

def print_top_words(topic_word):
	for i, topic_dist in enumerate(topic_word):
	    topic_words 	= np.array(tf_feature_names)[np.argsort(topic_dist)][:-(n_words+1):-1]
	    print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))

if __name__=='__main__':
	n_words 			= 15
	n_topics   			= 20
	n_top_words 		= 20

	stop_words 			= ['*']
	sms 				= pd.read_csv('/Users/Nickhil_Sethi/Documents/Datasets/smsspamcollection/SMSSpamCollection.csv').as_matrix()

	print("Extracting tf features for LDA...")
	tf_vectorizer       = CountVectorizer(stop_words=stop_words)
	t0                  = time()
	tf                  = tf_vectorizer.fit_transform(sms[:,2])
	tf_feature_names    = tf_vectorizer.get_feature_names()
	print("done in %0.3fs." % (time() - t0))

	# removed_rows		= []
	# new_tf 				= []
	# for idx, element in enumerate(tf):
	# 	if element.sum() == 0:
	# 		removed_rows.append(idx)
	# 	else:
	# 		new_tf.append(element)
	# print removed_rows

	if True:
		n_samples  			= tf.shape[0]
		n_features 			= len(tf_feature_names)
		
		print("Fitting LDA models with tf features, n_samples=%d and n_features=%d..." % (n_samples, n_features))
		model      			= lda.LDA(n_topics=n_topics)
		t0         			= time()
		model.fit(tf)
		print("done in %0.3fs." % (time() - t0))

		print("\nTopics in LDA model:")
		topic_word 			= model.topic_word_

		print model.doc_topic_