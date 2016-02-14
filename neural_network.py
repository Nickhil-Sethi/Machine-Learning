import sys
sys.path.insert(0,'/Library/Python/2.7/site-packages')
import tensorflow as tf
import numpy



data = numpy.array([[0,0,0,0,0]]).T

class Logistic_Regression(object):
	def __init__(self):
		self.x = tf.placeholder("float",shape=[5,1])
		self.W = tf.Variable(tf.random_normal([2,5], stddev = .1), name='W')
		self.b = tf.Variable(tf.random_normal([2,1],stddev = .1), name='b')
		self.p_y_given_x = tf.nn.softmax( tf.matmul(self.W,self.x) + self.b )

LR = Logistic_Regression()
y_ = tf.placeholder("float",shape =[2,1])

sess = tf.Session()
sess.run(tf.initialize_all_variables())
print sess.run(LR.W)
print sess.run(LR.b)
print sess.run(LR.p_y_given_x, feed_dict = {LR.x: data})