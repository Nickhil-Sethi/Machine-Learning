'''

@author: Nickhil-Sethi

'''

import multiprocessing
import numpy as np
import spam_numerical as nm


#initializing dictionaries to hold word counts
spam_word_counts = {}
ham_word_counts = {}


''' input parameters '''

#ignore these words
stop_words = []

special_strings = ['www','com','xxx','sex','co']

print "    ...loading data"
dataset = open('/Users/Nickhil_Sethi/Documents/Datasets/smsspamcollection/SMSSpamCollection')
print "\ndata loaded.\n\n    ...computing word counts."

spam_count = 0
ham_count = 0
for line in dataset:
	sentence = line.split('\t')
	if sentence[0] == 'spam':
		for word in sentence[1].split(' '):
			if not word in stop_words:
				if word in spam_word_counts:
					spam_word_counts[word] += 1
				else:
					spam_word_counts[word] = 1
		spam_count += 1
	
	if sentence[0] == 'ham':
		for word in sentence[1].split(' '):
			if not word in stop_words:
				if word in ham_word_counts:
					ham_word_counts[word] += 1
				else:
					ham_word_counts[word] = 1
		ham_count += 1

	if len(spam_word_counts) > 400:
		break



''' computing perplexities '''



# table[word] ::: 
#  'a' spam w/ word     |   'b' ham w/ word
#  'c' spam w/out word  |   'd' ham w/out word
table = {}

print "\n","    ...assembling tables"
for word in spam_word_counts:
	table[word] = {}
	table[word]['a'] = spam_word_counts[word]
	table[word]['c'] = spam_count - spam_word_counts[word]
	if word in ham_word_counts:
		table[word]['b'] = ham_word_counts[word]
		table[word]['d'] = ham_count - ham_word_counts[word]
	else:
		table[word]['b'] = 0
		table[word]['d'] = ham_count
for word in ham_word_counts:
	if not word in spam_word_counts:
		table[word] = {}
		table[word]['a'] = 0
		table[word]['c'] = spam_count
		table[word]['b'] = ham_word_counts[word]
		table[word]['d'] = ham_count - ham_word_counts[word]

print "    ...computing perplexities", "\n"

ls = float(len(spam_word_counts))

count = 0
check_in=50

spam_perplexities = {}
for word in spam_word_counts:

	spam_perplexities[word] = nm.fisher_exact(table[word])
	count += 1
	
	if count%check_in == 0:
		print "{}% of spam perplexities computed...".format( 100*float(count)/ls )

print "\n","    ...sorting words", "\n"

spam_sorted_words = nm.dictionary_insertion_sort(spam_perplexities)
for word in spam_sorted_words:
	if spam_perplexities[word] > .05:
		break
	w = word.encode('utf-8')
	print "spam perplexity of \'{}\' = {} ; count = {}".format(w,spam_perplexities[word],spam_word_counts[word])





