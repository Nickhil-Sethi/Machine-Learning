
from smapp_toolkit.twitter import MongoTweetCollection
from datetime import datetime, timedelta
from pprint import pprint
import numpy as np


#returns sum of logs of numbers between l and u (inclusive)
def log_sum(u,l):
	sum = 0
	for i in xrange(l,u+1):
		sum = sum + np.log(i)
	return sum

def compute_perplexity(table_word):

	a = table_word['a']
	b = table_word['b']
	c = table_word['c']
	d = table_word['d']
	
	l_sum = log_sum(a+b,b+1) + log_sum(c+d,c+1) + log_sum(a+c,a+1) + log_sum(b+d,d+1) - log_sum(a+b+c+d,1)
	return np.e**l_sum

collection = MongoTweetCollection(address='localhost',
                                  port=49999,
                                  username='smapp_readOnly',
                                  password='',
                                  dbname='NewYorkGeo',
                                  authentication_database='admin')
num_NY = collection.count()
collection2 = MongoTweetCollection(address='localhost',
                                  port=49999,
                                  username='smapp_readOnly',
                                  password='',
                                  dbname='SanFranciscoGeo',
                                  authentication_database='admin')
num_SF = collection2.count()

#finance? 
#look at 'the', to demonstrate a word for which there is no difference
pwords = ['swag','race','shit','sex', 'racist','Manhattan','San Francisco','Obama','cold','hot','times','bay','Trump']
pcounts = {}
pcounts2 = {}
for w in pwords:
	pcounts[w] = 0
	pcounts2[w] = 0

#check in every 55 tweets 
check_in = 55
#stop after this many tweets
stop = 1000

#selection probabiliy
r = .5

NY_count = 0
print 'r =', r
tweet_dict = {}
for tweet in collection:
	if tweet['random_number'] <= r:
		#store tweet; don't do this unless you have to
		#tweet_dict[tweet['id_str']] = tweet['text']
		#incrementing word counts
		for word in pwords:
			if word in tweet['text']:
				pcounts[word] += 1
		#incrementing tweet count
		NY_count += 1
		#check in
		if NY_count%check_in == 0:
			print "{}% of NY tweets processed...".format( 100*float(NY_count)/( stop ) )
		#stopping condition
	if stop and NY_count == stop:
		break

SF_count = 0
for tweet in collection2:
	if tweet['random_number'] <= r:
		#tweet_dict[tweet['id_str']] = tweet['text']
		for word in pwords:
			if word in tweet['text']:
				pcounts2[word] += 1
		SF_count += 1
		if SF_count%check_in == 0:
			print "{}% of SF tweets processed...".format( 100*float(SF_count)/( stop ) )
		if stop and SF_count == stop:
			break

table = {}
for word in pwords:
	table[word] = {}
	table[word]['a'] = pcounts[word]
	table[word]['b'] = pcounts2[word]
	table[word]['c'] = NY_count - pcounts[word]
	table[word]['d'] = SF_count - pcounts2[word]

print
for word in pwords:
	print "perplexity of {} = {}".format(word, compute_perplexity(table[word]))

print table['Trump']

print "NY ", pcounts
print "SF ", pcounts2