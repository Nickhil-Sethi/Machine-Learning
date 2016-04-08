'''


Dear Reader,

I wrote this code for a research project on geolocating tweets, when I was working at SMaPP Lab (this code is from the 
early stages of the project). I no longer have access to their data set, but if you have your own set of text data, 
feel free to use my code for yourself -- data just needs to be in a structure which is iterable.

-Nickhil



4th revision of tweet geolocation code.

This code searches for words that have predictive value for classifying tweets as from
New York (NY) or San Francisco (SF). Soon we'll add a large outgroup population (Mid-West) to simulate a more realistic dataset.

Predictive words are found through a version of Fisher's "Exact Test" (see the stirling.py module for implementation)
All words with p-value from Fisher Exact Test < .05 are reported in decreasing order of significance.

NOTE: Code can run on samples of < 800,000; otherwise computing p-value encounters numerical issues.

Results: running code on random samples indicates that some words have strong predictive value for geographic origin; 
unfortunately, no words are both predictive and common, so it's a difficult to use 
individual words to classify individual tweets. Might be useful to classify a corpus of tweets. 
For individual tweets, perhaps we can use LDA?

@author : Nickhil-Sethi

'''



import subprocess
import numpy as np
import stirling as st
from smapp_toolkit.twitter import MongoTweetCollection

# ssh to MongoDB Tweet collection


collection = MongoTweetCollection(address='localhost',
                                  port=49999,
                                  username='smapp_readOnly',
                                  password='',
                                  dbname='NewYorkGeo',
                                  authentication_database='admin')

collection2 = MongoTweetCollection(address='localhost',
                                  port=49999,
                                  username='smapp_readOnly',
                                  password='',
                                  dbname='SanFranciscoGeo',
                                  authentication_database='admin')

#counts
num_NY = collection.count()
num_SF = collection2.count()
total = float(num_NY + num_SF)

#percentages
percent_NY = float(num_NY)/float(total)
percent_SF = float(num_SF)/float(total)


#initializing dictionaries to hold word counts
NY_word_counts = {}
SF_word_counts = {}
other_word_counts = {}


''' input parameters '''

#ignore these words
stop_words = ['', '  ', '   ', ' ', '.', ',', 'his', 'are', 'but', 'for', 'you', 'our', \
'out', 'i', 'The', 'of', 'so','such','any','the','at','and','a','to','in',\
'not','he','she','it','they','we','I','me', 'this','This', 'than', '@', '\n']


#stop after this many tweets, or sample whole database
#takes value of positive integer or boolean 'False'
stop = True
#stopping conditions for sf,ny,mw; only matter if stop == True
sf_stop = 1500
ny_stop = 3000
mw_stop = 2000
#selection window; tweets with 'random_number' in this interval are selected
r_high = .9
r_low = .8
#collect data with random number meeting these criteria
collecting_condition = lambda x: (r_low < x < r_high)

#check in every # tweets 
check_in = 150




''' sanity checks '''
#check that selection window is non-degenerate
assert r_high > r_low
#check to see that 'stop' variable is either an integer or boolean, but not 'True'
if not isinstance(stop, bool):
	raise ValueError('stop must be bool')
#check that stop is positive
if stop == True:
	assert isinstance(sf_stop,int) and isinstance(ny_stop,int) and isinstance(mw_stop,int)
	assert sf_stop > 0 and ny_stop > 0 and mw_stop > 0
#check to see that we aren't collecting more data than we can handle
if  stop == False and r_high - r_low > .1:
	raise ValueError("warning: attempting to take more than 1% sample is not advised")


if stop == True:
	print "\n","Taking a random sample of {} NY tweets, {} SF tweets, {} MW tweets.".format(ny_stop,sf_stop,mw_stop)
else:
	print "\n","Taking an {}% sample of each database.".format(100*(r_high - r_low))





'''Processing New York Tweets'''

#processing NY tweets
NY_count = 0
#tweet_dict = {}
for tweet in collection:
	if collecting_condition(tweet['random_number']):
		
		#incrementing word counts		
		for word in tweet['text'].split(' '):
			#This step first because it is quick to confirm that word is stop
			if not word in stop_words:
				if word in NY_word_counts:
					NY_word_counts[word] += 1
				else:
					NY_word_counts[word] = 1
		#incrementing tweet count
		NY_count += 1
		#check in
		if NY_count%check_in == 0:
			if stop:
				print "{}% of expected NY tweets processed...".format( 100*float(NY_count)/float(ny_stop) )
			else:
				print "{}% of NY tweets processed...".format( 100*float(NY_count)/( num_NY ) )

		#stopping condition
	if stop and NY_count == ny_stop:
		break

print "\n"



'''Processing San Francisco Tweets'''

SF_count = 0
for tweet in collection2:
	if collecting_condition(tweet['random_number']):
		#incrementing wordcounts	
		for word in tweet['text'].split(' '):
			#This step first because it is quick to confirm that word is stop
			if not word in stop_words:
				if word in SF_word_counts:
					SF_word_counts[word] += 1
				else:
					SF_word_counts[word] = 1
		SF_count += 1
		if SF_count%check_in == 0:
			if stop:
				print "{}% of expected SF tweets processed...".format( 100*float(SF_count)/float(sf_stop) )
			else:
				print "{}% of SF tweets processed...".format( 100*float(SF_count)/( num_SF ) )


		if stop and SF_count == sf_stop:
			break



''' computing perplexities '''


table = {}

print "\n","    ...assembling tables"
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

print "    ...computing perplexities", "\n"

NY_perplexities = {}
count = 0
for word in NY_word_counts:
	NY_perplexities[word] = st.fisher_exact(table[word])
	count += 1
	if count%check_in == 0:
		print "{}% of NY perplexities computed...".format( 100*float(count)/float(len(NY_word_counts)) )

print "\n","    ...sorting words", "\n"

NY_sorted_words = st.dictionary_bubble_sort(NY_perplexities)
for word in NY_sorted_words:
	if NY_perplexities[word] > .05:
		break
	print "NY perplexity of {} = {} ; count = {}".format(word.encode('utf-8'),NY_perplexities[word],NY_word_counts[word])
