
import tensorflow as tf
from __future__ import print_function


v=tf.Variable(0.0,name='firstvar')

one=tf.constant(1.0)
increment=tf.assign(v,v+one)


sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

for i in range(5):
	val,_=sess.run([v,increment])
	print('i=',i,'v=',val)
sess.close()

##how to modify to output sum of n squares???
