from __future__ import print_function
import tensorflow as tf



v=tf.Variable(0.0,name='firstvar')

step=tf.placeholder(dtype='float',shape=[],name='step')
sq=tf.square(step)
increment=tf.assign(v,v+sq)


sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

#now suppose you got n from some intermediate step after some computation
n=5 
for i in range(n):
	val=0
	val,_=sess.run([v,increment],feed_dict={step:i})

print('v=',val)

sess.close()



