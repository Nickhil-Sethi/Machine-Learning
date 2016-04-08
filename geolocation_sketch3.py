
from smapp_toolkit.twitter import MongoTweetCollection
import stirling as st
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

stop_words = ['  ', '   ' ' ', 'his', 'are', 'but', 'for', 'you', 'our', 'out', 'i', 'The', 'so','such','any','the','at','and','a','to','in',\
'not','he','she','it','they','we','I','me', '\n']

NY_word_counts = {}
SF_word_counts = {}

#check in every 55 tweets 
check_in = 55
#stop after this many tweets
stop = None
#selection probabiliy
r = .005

print "taking a {}% sample of the {} NY tweets and {} SF tweets.".format(100*r,num_NY,num_SF)

NY_count = 0
tweet_dict = {}
for tweet in collection:
	if tweet['random_number'] <= r:
		#store tweet; don't do this unless you have to
		#tweet_dict[tweet['id_str']] = tweet['text']
		
		#incrementing word counts		
		for word in tweet['text'].split(' '):
			#This step first because it is quick to confirm that word is stop
			#print word.encode('utf-8')
			if not word in stop_words:
				if word in NY_word_counts:
					NY_word_counts[word] += 1
				else:
					NY_word_counts[word] = 1
		#incrementing tweet count
		NY_count += 1
		#check in
		if NY_count%check_in == 0:
			print "{}% of NY tweets processed...".format( 100*float(NY_count)/( num_NY ) )
		#stopping condition
	if stop and NY_count == stop:
		break

SF_count = 0
for tweet in collection2:
	if tweet['random_number'] <= r:
		#tweet_dict[tweet['id_str']] = tweet['text']
				#incrementing word counts		
		for word in tweet['text'].split(' '):
			#This step first because it is quick to confirm that word is stop
			if not word in stop_words:
				if word in SF_word_counts:
					SF_word_counts[word] += 1
				else:
					SF_word_counts[word] = 1
		SF_count += 1
		if SF_count%check_in == 0:
			print "{}% of SF tweets processed...".format( 100*float(SF_count)/( num_SF ) )	
		if stop and SF_count == stop:
			break

table = {}

for word in NY_word_counts:
	table[word] = {}
	table[word]['a'] = NY_word_counts[word]
	table[word]['c'] = NY_count - NY_word_counts[word]
	if word in SF_word_counts:
		table[word]['b'] = SF_word_counts[word]
		table[word]['d'] = SF_count - SF_word_counts[word]
	else:
		table[word]['b'] = 0
		table[word]['d'] = SF_count
for word in SF_word_counts:
	if not word in NY_word_counts:
		table[word] = {}
		table[word]['a'] = 0
		table[word]['c'] = NY_count
		table[word]['b'] = SF_word_counts[word]
		table[word]['d'] = SF_count - SF_word_counts[word]
print "\n"
NY_perplexities = {}
count = 0
for word in NY_word_counts:
	NY_perplexities[word] = compute_perplexity(table[word])
	count += 1
	if count%check_in == 0:
		print "{}% of NY perplexities computed...".format( 100*float(count)/float(len(NY_word_counts)) )
SF_perplexities = {}
count = 0
for word in SF_word_counts:
	SF_perplexities[word] = compute_perplexity(table[word])
	count += 1
	if count%check_in == 0:
		print "{}% of SF perplexities computed...".format( 100*float(count)/float(len(SF_word_counts)) )
print "\n"
for word in NY_word_counts:
	if NY_perplexities[word] < .05:
		print "NY perplexity of {} = {}".format(word.encode('utf-8'),NY_perplexities[word])
for word in SF_word_counts:
	if SF_perplexities[word] < .05:
		print "SF perplexity of {} = {}".format(word.encode('utf-8'),SF_perplexities[word])
