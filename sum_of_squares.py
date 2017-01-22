
import tensorflow as tf
from __future__ import print_function

##How do you fix this answer if you don't know n at graph construction??
n=5


v=tf.Variable(0.0,name='firstvar')

for step in range(n):
	sq=tf.square(tf.constant(step,'float'))
	#increment=tf.assign(v,v+sq)
	v=v+sq

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

val=sess.run(v)

print('v=',val)

sess.close()



