import string
import numpy as np 

N 			= 4
scale 		= 50
T 			= 1000
sim_time 	= 1000


initial 	= scale*np.random.rand(N)
transition  = [ np.random.dirichlet([1 for x in xrange(N)]) for y in xrange(N)] 
transition  = np.array(transition).T

c 	= 0
while c < sim_time:
	c+=1
	initial = transition.dot(initial) + np.random.normal(N)
	if c%10==0:
		print(initial)

