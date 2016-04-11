import numpy as np
import scipy.stats as stats
from copy import copy


"""A library of special numerical methods and approximations for tweet geolocation project"""

# stirling's approximation to n!
def stirling(n):
	return float( np.sqrt(2*np.pi*n)*( (float(n)/np.e)**n) )

# sum of log(n) for n in [1,u], inclusive
def log_sum(u,l=1):
	return sum([np.log(i) for i in xrange(l,u+1)])

# more numerically stable and efficient way to compute perplexity, e.g. hypergeometric distribution
def compute_perplexity(table_word):
	a = table_word['a']
	b = table_word['b']
	c = table_word['c']
	d = table_word['d']
	
	l_sum = log_sum(a+b,b+1) + log_sum(c+d,c+1) + log_sum(a+c,a+1) + log_sum(b+d,d+1) - log_sum(a+b+c+d,1)
	return np.e**l_sum

# computes perplexity of a word and integrates over tails
# we have to do this a lot for large corpuses --
# can we just lookup previous answers if already computed?
def fisher_exact(table_word):

	a = table_word['a']
	b = table_word['b']
	c = table_word['c']
	d = table_word['d']

	if a >= c:
		sum = 0
		for i in xrange(c+1):
			ai = a+i
			ci = c-i
			table = {'a':ai, 'b':b , 'c':ci,'d':d}
			sum = sum + compute_perplexity(table)
		return sum
	else:
		sum = 0
		for i in xrange(a+1):
			ai = a-i
			ci = c+i
			table = {'a':ai, 'b':b, 'c':ci, 'd':d}
			sum = sum + compute_perplexity(table)
		return sum

# returns list of keys in order of increasing dictionary values
# does not return a full dictionary because of high overhead
def dictionary_bubble_sort(dictionary):
	if not isinstance(dictionary,dict):
		raise TypeError('bubble sort must receive dictionary')
	ks = list(dictionary.keys())
	num_keys = len(ks)
	swap = True
	while(swap):
		swap = False
		for i in xrange(num_keys-1):
			# if value of current key is greater than next key 
			if dictionary[ks[i]] > dictionary[ks[i+1]]:
				#swap
				temp = copy(ks[i])
				ks[i] = ks[i+1]
				ks[i+1] = temp
				swap = True		
	return ks

if __name__ == '__main__':
	print log_sum(5,1)

	import string
	import numpy.random as rng

	d = {}
	count = 1
	for l in string.letters:
		# signal + noise
		d[l] = .1*count+ .5*rng.rand()
		count += 1

	sorted_keys = dictionary_bubble_sort(d)

	for i in sorted_keys:
		print i,d[i]

	def dictionary_test(sorted_keys,dictionary):
		for i in xrange(len(sorted_keys)-1):
			if dictionary[sorted_keys[i]] > dictionary[sorted_keys[i+1]]:
				print "dictionary not well sorted"
				return True

		print "dictionary well sorted"
		return False

	print "\n"

	dictionary_test(d.keys(),d)
	dictionary_test(sorted_keys,d)
