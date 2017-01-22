import numpy as np
import time,os
from sklearn.metrics import roc_curve, auc

import tensorflow as tf
import tensorflow as tf
#from tensorflow.models.rnn import rnn, rnn_cell
#from tensorflow.models.rnn.rnn_cell import RNNCell

try:#version 0.9
    GRUCell=tf.nn.rnn_cell.GRUCell
    BasicLSTMCell=tf.nn.rnn_cell.BasicLSTMCell
except:#0.8 and below
    from tensorflow.models.rnn.rnn_cell import GRUCell, BasicLSTMCell

from tensorflow.python.ops import math_ops,init_ops
from tensorflow.python.ops import variable_scope as vs

#tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
#tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
#                          "Learning rate decays by this much.")

from tensorflow.python.framework import errors
def _filter_exception(ex):
    if isinstance(ex,tuple):
        ex2=ex[1]
    else:
        ex2=ex
    if isinstance(ex2,(errors.OutOfRangeError)):
        print 'Out of Range exception'
        ex=None
    return ex

from RecordReader import DataStream
from sequential_graph import RNNGraph
class RNNModel(object):
	#def run(target,inputs):
	def init_variables(self):
            #self.coord = tf.train.Coordinator()
            #self.session.run(tf.trainable_variables())
            all_vars=tf.all_variables()
            uninit_names=self.session.run( tf.report_uninitialized_variables() )
            self.init_vars= filter( lambda v: v.name.rsplit(':',1)[0] in uninit_names , all_vars )
            #self.init_vars= self.session.run( tf.report_uninitialized_variables() )
            self.session.run( tf.initialize_variables( self.init_vars ) )
            #self.session.run(tf.initialize_all_variables())
	def load_model(self):
            if not os.path.isdir(self.model_folder):
                assert(not os.path.exists(self.model_folder))
                os.mkdir(self.model_folder)
             #self.model_name=model_name
            #self.model_folder= os.path.join(self.model_dir,self.model_name)
            ckpt = tf.train.get_checkpoint_state(self.model_folder)
            if ckpt:
                self.saver.restore(self.session, ckpt.model_checkpoint_path)
                print 'Loading Model'

	def __init__(self,session, cell,uses_context=False,train_inputs=None, test_inputs=None,n_input=None, scope='rnn_model'):
		self.session=session
		flags = tf.app.flags
		FLAGS = flags.FLAGS
		display_step = FLAGS.display_step
		self.display_step= FLAGS.display_step
		learning_rate= FLAGS.learning_rate
		self.uses_context=uses_context
		self.model_dir=FLAGS.model_dir
		self.n_context=FLAGS.n_context
		self.n_steps = FLAGS.n_steps# timesteps [some sequences truncated]
		self.n_hidden= FLAGS.n_hidden
		self.n_classes=2
		self.mystep = 0
		self.model_folder= os.path.join(self.model_dir,scope)
		self.num_threads=FLAGS.num_threads
		self.batch_size=FLAGS.batch_size
		self.training_iters=FLAGS.training_iters

                #self.global_step=tf.Variable( 0 , name='global_step',trainable=False)
                self.global_step=tf.get_variable('global_step',shape=[],initializer=tf.constant_initializer(1),trainable=False)
                ###Use with count_up_to later:
                self.training_iters=FLAGS.training_iters

		with tf.variable_scope(scope,reuse=None):
                    #maybe uneccesary to pass scope as arg
                    self.training_graph= RNNGraph( session,cell, uses_context,connected_inputs=train_inputs,n_input=n_input,is_training=True,global_step=self.global_step,scope=scope)

		with tf.variable_scope(scope,reuse=True):
			self.testing_graph= RNNGraph( session, cell,uses_context,connected_inputs=test_inputs,n_input=n_input,is_training=False,global_step=self.global_step,scope=scope)

            #with tf.device('/cpu:0'):
                #self.saver=tf.train.Saver( [self.global_step]  )
                self.save_vars=tf.trainable_variables()
                self.save_vars.append(self.global_step)
                self.saver=tf.train.Saver( self.save_vars )

	#	self.saver=tf.train.Saver()

        def save(self):
            #step=self.session.run(self.global_step)
            #step=self.session.run(self.training_graph.global_step)
            save_path = self.saver.save(self.session,self.model_folder+ "/rnn.ckpt",global_step=self.global_step)
            #save_path = self.saver.save(self.session, self.model_folder+ "/rnn.ckpt",step)
            print("Model saved in file: %s" % save_path)

	def fit(self,inputs=None):
		sess=self.session
		#sess.run( tf.initialize_all_variables())
		coord = tf.train.Coordinator() #coord=self.coord
		threads=tf.train.start_queue_runners(coord=coord,sess=sess)
		i=1
		try:
                    t1=time.time()
                    while not coord.should_stop():
                        #if self.mystep>=self.training_iters:
                        #        print 'training_iters exceeded'
                        #        coord.request_stop()
                        self.step(inputs)
                        i+=1
		except Exception, e:
                        e=_filter_exception(e)#in master but not 0.8
			t2=time.time()
			print 'num_threads',self.num_threads
			print 'batch_size',self.batch_size
			print 'n iter=',i
			print 'Time per Step:',(t2-t1)/i
			print 'Total Time:',t2-t1#,'\n\t',t1-t0
			coord.request_stop( e )
		finally:
			coord.request_stop()
			coord.join(threads, stop_grace_period_secs=100)
                        self.save()

	def step(self,inputs=None):
            if inputs is None:
                fd=None
            else:
                raise NotImplementedError
                fd=some_stuff
            _,step=self.session.run([self.training_graph.train_op,self.global_step], feed_dict=fd)
            self.mystep +=1
            #self.training_graph.write(fd =fd)
            if self.mystep % self.display_step == 0:
                self.save()
                acc,loss=self.training_graph.write(fd)
                print "Step " + str(np.round(step)) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                          ", Training Accuracy= " + "{:.5f}".format(acc)

	def test_eval(self,inputs=None):
		if inputs is None:
			fd=None
		else:
			raise NotImplementedError
		sess=self.session
                #should already be initialized
		coord = tf.train.Coordinator() #coord=self.coord
		threads=tf.train.start_queue_runners(coord=coord,sess=sess)
		i=0
		total_count=0
		total_correct=0
		P=[];A=[]

                print 'Starting testing at global_step ',sess.run(self.global_step)
		coord.clear_stop()
		try:#with coord.stop_on_exception():#should join at end
                        t1=time.time()
			while not coord.should_stop():
				i+=1
				n,n_c,np_y_,np_y_hat_=sess.run( self.testing_graph.testing_targets)
				total_count+=n
				total_correct+=n_c
				P.append(np_y_hat_[:,0])
				A.append(np_y_[:,0])
		except Exception, e:
                        e=_filter_exception(e)#in master but not 0.8
                        t2=time.time()
			print 'Finished Testing: n iter=',i
			coord.request_stop( e )
			print 'Total Time:',t2-t1
		finally:
			coord.request_stop()
			coord.join(threads, stop_grace_period_secs=100)
		npA=np.concatenate(A)
		npP=np.concatenate(P)
		mask=np.where(npA>-1)[0]
		actual,predictions=npA[mask],npP[mask]
                if total_count>0:
                    print 'Test Accuracy: ', float(total_correct)/total_count
                else:
                    print 'total_count is zero?!'
		false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
		roc_auc = auc(false_positive_rate, true_positive_rate)
		print 'AUC: ', roc_auc
		#return actual, predictions, total_count, total_correct



