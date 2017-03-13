"""Experiments with Latent Dirichlet Allocation for compressing and detecting spam texts"""

from time import time

import lda
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

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

	sms 				= pd.read_csv('/Users/Nickhil_Sethi/Documents/Datasets/smsspamcollection/SMSSpamCollection.csv').as_matrix()
	sms 				= sms[sms['test'].split(' ') != '']

	print("Extracting tf features for LDA...")
	tf_vectorizer       = CountVectorizer(max_df=0.95, min_df=4, stop_words='english')
	t0                  = time()
	tf                  = tf_vectorizer.fit_transform(sms[:,2])
	tf_feature_names    = tf_vectorizer.get_feature_names()
	print("done in %0.3fs." % (time() - t0))

	n_samples  			= len(tf)
	n_features 			= len(tf_feature_names)
	
	print("Fitting LDA models with tf features, n_samples=%d and n_features=%d..." % (n_samples, n_features))
	model      			= lda.LDA(n_topics=n_topics)
	t0         			= time()
	model.fit(tf)
	print("done in %0.3fs." % (time() - t0))

	print("\nTopics in LDA model:")
	topic_word 			= model.topic_word_

	print model.doc_topic_