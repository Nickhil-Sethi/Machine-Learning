"""
=======================================================================================
Topic extraction with Non-negative Matrix Factorization and Latent Dirichlet Allocation
=======================================================================================

This is an example of applying Non-negative Matrix Factorization
and Latent Dirichlet Allocation on a corpus of documents and
extract additive models of the topic structure of the corpus.
The output is a list of topics, each represented as a list of terms
(weights are not shown).

The default parameters (n_samples / n_features / n_topics) should make
the example runnable in a couple of tens of seconds. You can try to
increase the dimensions of the problem, but be aware that the time
complexity is polynomial in NMF. In LDA, the time complexity is
proportional to (n_samples * iterations).
"""


# from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

import pandas as pd

# rewrite this; taken from internet
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]) )
    print()

sms = pd.read_csv('/Users/Nickhil_Sethi/Documents/Datasets/smsspamcollection/SMSSpamCollection.csv')

print("Extracting tf features for LDA...")
tf_vectorizer       = CountVectorizer(max_df=0.95, min_df=4, stop_words='english')
t0                  = time()
tf                  = tf_vectorizer.fit_transform(sms['text'].as_matrix())
tf_feature_names    = tf_vectorizer.get_feature_names()
print("done in %0.3fs." % (time() - t0))

n_samples  = len(sms)
n_features = len(tf_feature_names)
print("Fitting LDA models with tf features, n_samples=%d and n_features=%d..." % (n_samples, n_features))

n_topics   = 20
lda        = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,learning_method='online', learning_offset=50.,random_state=0)
t0         = time()
m          = lda.fit_transform(tf)
print("done in %0.3fs." % (time() - t0))

print("\nTopics in LDA model:")
n_top_words = 20
print_top_words(lda, tf_feature_names, n_top_words)

for line in m:
    print line
print m.shape