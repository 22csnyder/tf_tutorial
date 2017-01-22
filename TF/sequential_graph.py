import numpy as np
import time,os

import tensorflow as tf
#from tensorflow.models.rnn import rnn, rnn_cell
try:#version 0.9
    GRUCell=tf.nn.rnn_cell.GRUCell
    BasicLSTMCell=tf.nn.rnn_cell.BasicLSTMCell
    RNNCell= tf.nn.rnn_cell.RNNCell
    rnn_cell=tf.nn.rnn_cell
except:#0.8 and below
    from tensorflow.models.rnn.rnn_cell import GRUCell, BasicLSTMCell
    from tensorflow.models.rnn.rnn_cell import RNNCell


from tensorflow.python.ops import math_ops,init_ops
from tensorflow.python.ops import variable_scope as vs

#tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
#tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
#                          "Learning rate decays by this much.")

def _simple_tanh_layer(inputs,output_size=None,scope=None):
	#output_size=position.get_shape()[1]
	input_size=inputs.get_shape()[0]
	if output_size is None:
		output_size=input_size

	with vs.variable_scope(scope or "NN layer"):
		W = vs.get_variable("W", [input_size,output_size],
			#initializer=init_ops.constant_initializer(1.0))
			initializer=init_ops.random_normal_initializer(mean=0.0, stddev=1.0))
		b=vs.get_variable('b', [1,output_size],
			initializer=init_ops.random_normal_initializer(mean=0.0, stddev=1.0))
		enc=tf.matmul(inputs,W)+b
		return tf.tanh( enc )

class TimeWrapper(RNNCell):
	def __init__(self,cell,both=False):
		"""Create a cell that predicts evolution of state"""
		if not isinstance(cell, RNNCell):
			raise TypeError("The parameter cell is not a RNNCell.")
		self._output_size=cell.output_size#Hopefully even
		if self.output_size < 1:
			raise ValueError("Parameter output_size must be > 0: %d." % output_size)
		self._cell=cell
	@property
	def state_size(self):
		return self._cell.state_size

	@property
	def output_size(self):
		return self._output_size

	def __call__(self,inputs,state,scope=None):
		#top row is time gap for each batch
		#_t=tf.reshape( tf.slice(inputs,begin=[0,0],size=[-1,1]), [-1])	
		_t= tf.slice(inputs,begin=[0,0],size=[-1,1])
		_x=tf.slice(inputs,begin=[0,1],size=[-1,-1])#batch_size x state_size
		if both==False:
			c,h= tf.split(1,2,state)
			h=_simple_layer( tf.concat(1,[_t,h]), output_size= self.output_size,scope='TMh')
			state=tf.concat(1,[c,h])

		if both==True:
			state=_simple_layer( tf.concat(1,[_t,state]),output_size= self.state_size,scope='TMhc')
		return self._cell(inputs,state)

def _time_wrapper( delta_T, position, velocity, bias=True, bias_start=0.0, scope=None):
	output_size=position.get_shape()[1]
	with vs.variable_scope(scope or "time model"):
		time= tf.mul(delta_T,tau)
		if bias:
			bias_term = vs.get_variable(
				"b", [1,output_size],
				initializer=init_ops.constant_initializer(bias_start))
			time = tf.add(time,-bias_term)
	position= position + (tf.sigmoid( time ) * velocity)
	return position, velocity


		#_t= tf.slice(inputs,begin=[0,0],size=[-1,1])
		#_x=tf.slice(inputs,begin=[0,1],size=[-1,-1])#batch_size x state_size

def _trajectory_model(delta_T, position, velocity, bias=True, bias_start=0.0, scope=None):
	output_size=position.get_shape()[1]
	# Now the computation.
	with vs.variable_scope(scope or "Trajectory"):
		##Try making tau a variable of inputs too
		tau = vs.get_variable("Tau", [1,output_size],
			#initializer=init_ops.constant_initializer(1.0))
			initializer=init_ops.random_normal_initializer(mean=1.0, stddev=1.0))
	#	res= math_ops.mul(tau, delta_T
		#batch_sizexstate_size  *elem-wise* 1xstate_size
		time= tf.mul(delta_T,tau)
		if bias:
			bias_term = vs.get_variable(
				"lag", [1,output_size],
				initializer=init_ops.constant_initializer(bias_start))
			time = tf.add(time,-bias_term)
	position= position + (tf.sigmoid( time ) * velocity)
	return position, velocity

def tf_eye(n,dtype=tf.float32):
	return tf.constant(np.eye(n),dtype=dtype)

def perm_mat(n):
	if n%4 != 0:
		raise ValueError('vector must have even length for half to be seen as a velocity')
	m=np.int(n/4)
	e=tf_eye(m)
	z=tf.zeros([m,m],dtype=tf.float32)
	P=tf.concat(1, [tf.concat(0,[e,z,z,z]),tf.concat(0,[z,z,e,z]),tf.concat(0,[z,e,z,z]),tf.concat(0,[z,z,z,e])] )
	return P

#Think about using h for velocity, c for state
class TrajectoryWrapper(RNNCell):
	def __init__(self,cell):
		"""Create a cell that predicts evolution of state"""
		if not isinstance(cell, RNNCell):
			raise TypeError("The parameter cell is not a RNNCell.")
		self._output_size=cell.output_size#Hopefully even
		if self.output_size < 1:
			raise ValueError("Parameter output_size must be > 0: %d." % output_size)
		self._cell=cell
		self.perm_mat= perm_mat(self.state_size) #2*n_hidden x 2*n_hidden
	@property
	def state_size(self):
		return self._cell.state_size

	@property
	def output_size(self):
		return self._output_size

	def __call__(self,inputs,state,scope=None):
		#top row is time gap for each batch
		#_t=tf.reshape( tf.slice(inputs,begin=[0,0],size=[-1,1]), [-1])	
		_t= tf.slice(inputs,begin=[0,0],size=[-1,1])
		_x=tf.slice(inputs,begin=[0,1],size=[-1,-1])#batch_size x state_size

		#tf.matmul( _x, self.perm_mat)
		s=tf.matmul( state, self.perm_mat)
                pos,vel=tf.split(1,2, s )#2x( batch_size x n_hidden )
		pos,vel=_trajectory_model( _t, pos, vel)
		state= tf.matmul( tf.concat(1,[pos,vel]) ,self.perm_mat)
		return self._cell(_x,state)


class RNNGraph(object):
	#def run(target,inputs):
	#def __init__(self,n_input,n_steps,n_hidden,model_name,cell,n_context=None):
    def __init__(self,session, cell,  uses_context=False, connected_inputs=None, n_input=None,is_training=False,global_step=None,scope='rnn_graph'):
        if global_step is None:
            raise ValueError('global step should be a tf variable')
        self.global_step = global_step
        self.session=session
        self.is_training=is_training
        flags = tf.app.flags
        FLAGS = flags.FLAGS
        display_step = FLAGS.display_step
        n_hidden= FLAGS.n_hidden
        self.display_step= FLAGS.display_step
        learning_rate= FLAGS.learning_rate
        self.uses_context=uses_context
        self.batch_size= FLAGS.batch_size
        self.model_dir=FLAGS.model_dir

        self.n_context=FLAGS.n_context
        self.n_steps = FLAGS.n_steps# timesteps [some sequences truncated]
        self.n_hidden= FLAGS.n_hidden
        self.n_input = n_input or FLAGS.n_input
        self.n_classes=2


        ###Feed dict###
        ##Placeholders##
        self.x = tf.placeholder_with_default(connected_inputs[0], [None,self.n_steps, FLAGS.n_input]) #[batch_size,n_steps, n_input]
        self.y = tf.placeholder_with_default(connected_inputs[1], [None, self.n_steps, self.n_classes])#[batch_size, n_steps, self.n_classes]
        self.early_stop = tf.placeholder_with_default(connected_inputs[2], [None]) #[batch_size]
        self.mask=tf.placeholder_with_default(connected_inputs[3],[None,self.n_steps]) #n_steps,batch_size ##Used to be transposed when fed in
        self.p_vec=tf.placeholder_with_default(connected_inputs[4],[None,self.n_context])
        self.delta_t= tf.placeholder_with_default(connected_inputs[5],[None,self.n_steps])


        if self.uses_context:
            self.istate= _simple_tanh_layer( self.p_vec, 2*n_hidden , scope='W_context')
        else:
            #self.istate=tf.random_normal( (self.batch_size, 2*self.n_hidden), stddev=0.65,name='initial_state' )
            self.istate=tf.zeros( (self.batch_size, 2*self.n_hidden),name='initial_state')

        ##Time Model that uses elapsed time to event as feature[0]
        if self.n_input== FLAGS.n_input+1:
            t_=tf.expand_dims( input= self.delta_t, dim=2)
            self.x= tf.concat( concat_dim=2, values=[t_, self.x])

        #		###Feed dict###
        #		if not connected_inputs:
        #			raise NotImplementedError('Warning feed_dict method not fully implemented!!')
        #			##Placeholders##
        #			self.x = tf.placeholder("float", [None,self.n_steps, self.n_input]) #[batch_size,n_steps, n_input]
        #			self.istate = tf.placeholder("float", [None, 2*self.n_hidden]) #state & cell => 2x self.n_hidden
        #			self.y = tf.placeholder("float", [None, self.n_steps, self.n_classes])#[batch_size, n_steps, self.n_classes]
        #			self.early_stop = tf.placeholder(tf.int32, [None]) #[batch_size]
        #			self.mask=tf.placeholder('bool',[None,self.n_steps]) #n_steps,batch_size ##Used to be transposed when fed in
        #			####
        #			if self.uses_context:
        #				self.delta_t= tf.placeholder("float",[None,self.n_steps])
        #				self.p_vec=tf.placeholder("float",[None,self.n_context])
        #
        #		###Connected input through TFRecords###
        #		else:
        #			#self.x, self.y, self.early_stop, self.mask=connected_inputs[:4]
        #			self.x, self.y, self.early_stop, self.mask, self.p_vec, self.delta_t =connected_inputs
        #			
        #			if self.uses_context:
        #				self.istate= _simple_tanh_layer( self.p_vec, 2*n_hidden , scope='W_context')	
        #			else:
        #				#self.istate=tf.random_normal( (self.batch_size, 2*self.n_hidden), stddev=0.65,name='initial_state' )
        #				self.istate=tf.zeros( (self.batch_size, 2*self.n_hidden),name='initial_state')
        #
        #			if self.n_input== FLAGS.n_input+1:
        #				t_=tf.expand_dims( input= self.delta_t, dim=2)
        #				self.x= tf.concat( concat_dim=2, values=[t_, self.x])


        #			with tf.name_scope("t0_state"):
        #				M=tf.Variable(tf.random_normal([self.n_context,2*n_hidden]),name="M_t0")
        #				b=tf.Variable(tf.random_normal([ 2*n_hidden ]),name="b_t0")
        #				enc=tf.matmul(self.p_vec, M ) + b
        #				self.istate = tf.tanh( enc )#	
        #		else:
            #self.istate=tf.constant( np.random.normal(loc=0.0, scale=1.0, size=(2, 2)).astype(np.float32)) 


        #learning_rate = 0.001#https://github.com/aymericdamien/TensorFlow-Examples/
        #self.display_step=100
        #self.n_input=n_input
        #self.n_steps=n_steps
        #self.n_context=n_context
        #self.n_hidden =n_hidden #256 #32#64 # hidden layer num of features

        self.model_name=scope
        self.model_folder= os.path.join(self.model_dir,self.model_name)
        #tf.reset_default_graph()
        #self.training_iters=1000000

        self.n_classes = 2 #Readmitted<30 or not
        #self.global_step=tf.constant(0, 

        def random_uniform():
            ##Fix me later
            return tf.random_uniform_initializer(-1,1)

        self.mask=tf.transpose(self.mask,[1,0])

        with tf.variable_scope('embedding'):
            self.weights = {
                'hidden': tf.get_variable('W_h',shape=[self.n_input, self.n_hidden], initializer=random_uniform()), 
                'out': tf.get_variable('W_out',shape=[self.n_hidden, self.n_classes], initializer=random_uniform() )
                }
            self.biases = {
                'hidden': tf.get_variable('b_h',shape=[self.n_hidden], initializer=random_uniform()),
                'out': tf.get_variable('b_out',shape=[self.n_classes], initializer=random_uniform())
            }


        #with tf.variable_scope('embedding'):
        #	self.weights = {
        #		'hidden': tf.get_variable('W_h',[self.n_input, self.n_hidden], tf.random_normal([self.n_input, self.n_hidden])), 
        #		'out': tf.get_variable('W_out',[self.n_hidden, self.n_classes], tf.random_normal([self.n_hidden, self.n_classes]))
        #		}
        #	self.biases = {
        #		'hidden': tf.get_variable('b_h',[self.n_hidden], tf.random_normal([self.n_hidden])),
        #		'out': tf.get_variable('b_out',[self.n_classes], tf.random_normal([self.n_classes]))
        #	}


        if cell=='GRU':
            self.cell=rnn_cell.GRUCell( 2*self.n_hidden )
        elif cell=='BasicLSTM':
            self.cell = rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0)

        elif cell=='LSTM':
            print 'Not implemented'
        #			 cell = rnn_cell.LSTMCell( self.n_hidden, forget_bias=1.0)
        #				 (self, num_units, input_size=None,
        #						use_peepholes=False, cell_clip=None,
        #						initializer=None, num_proj=None,
        #						num_unit_shards=1, num_proj_shards=1,
        #						forget_bias=1.0, state_is_tuple=False)
        #
        elif not isinstance(cell,str):
            self.cell=cell #Assume it is an instance of an RNNCell model
        else:
            print 'Error: unrecognized rnn cell type specified'


        ###Define Penalty and Optimize
        self.pred_y2d = self.RNN(self.cell,self.x, self.istate, 
                            self.weights, self.biases, 
                            self.early_stop,self.delta_t)#( n_steps x batch_size, self.n_classes)

        self.class_probs=tf.nn.softmax(self.pred_y2d)


        # Evaluate model
        _bm=tf.reshape(self.mask,[-1])
        ppp=tf.argmax( self.pred_y2d, 1)
        y_tp=tf.transpose(self.y, [1, 0, 2])#batch_size,n_steps,self.n_classes
        self.y_2d=tf.reshape(y_tp,[-1,self.n_classes])#n_stepsxbatch_size,self.n_classes
        yyy=tf.argmax( self.y_2d, 1)
        eqls=tf.equal(ppp,yyy)#need to find less stupid variable names
        correct_pred= tf.logical_and(eqls,_bm)
        self.num_correct=tf.reduce_sum( tf.cast( correct_pred,tf.float32)  )
        self.num_possible=tf.reduce_sum(tf.cast( _bm,tf.float32 ) )
        self.accuracy=tf.div( tf.cast(self.num_correct,tf.float32), tf.cast(self.num_possible,tf.float32) )

        self.bias_out_summary = tf.scalar_summary(['b0','b1'], self.biases['out'])
        self.accuracy_summary = tf.scalar_summary("accuracy", self.accuracy)

        self.testing_targets=[self.num_possible, self.num_correct, self.y_2d, self.class_probs]
        #self.saver=tf.train.Saver(tf.all_variables())
        #Look at cost:
        if self.is_training:

            flat_mask=tf.reshape(tf.cast(self.mask,tf.float32),(-1,1)) #(n_steps x batch_size,1)
            count=tf.reduce_sum( flat_mask )
            averaging_mask=tf.div( flat_mask , count )
            # averaging_mask=tf.div( flat_mask , tf.cast( tf.reduce_sum(flat_mask), tf.float32 ) )

            #x = logits, z = targets
            #softmax[i, j] = exp(logits[i, j]) / sum(exp(logits[i]))
            # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            #exclusive labels. Expects logits
            ce=tf.nn.softmax_cross_entropy_with_logits(self.pred_y2d, self.y_2d) # (n_steps x batch_size,)

            ce_2d=tf.reshape(ce,(1,-1))
            # Define loss and optimizer
            # cost = tf.reduce_mean(ce) # Softmax loss
            self.cost=tf.squeeze( tf.matmul(ce_2d,averaging_mask) )

            self.train_op= tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost,global_step=self.global_step) # Adam Optimizer

            self.cost_summary=tf.scalar_summary('cost',self.cost)

            # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            #all_vars = tf.all_variables()
            #model_one_vars = [k for k in all_vars if k.name.startswith(FLAGS.model_one_scope)]
            #model_two_vars = [k for k in all_vars if k.name.startswith(FLAGS.model_two_scope)]
            #j_pq_vars    = [k for k in all_vars if k.name.startswith('j_pq')]
            #tf.train.Saver(model_one_vars).restore(sess, model_one_checkpoint)
            #tf.train.Saver(model_two_vars).restore(sess, model_two_checkpoint)

    def RNN(self,cell,_X, _istate, _weights, _biases, early_stop,delta_t):
        _X = tf.transpose(_X, [1, 0, 2])#(batch_size, n_steps, n_input)  # permute n_steps and batch_size
        _X = tf.reshape(_X, [-1, self.n_input]) # (n_steps*batch_size, n_input)
        #matmul(   (n_steps*batch_size, n_input) ,  (n_input, self.n_hidden)   )
        _X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']  #(n_steps*batch_size, self.n_hidden)

        if delta_t is not None:
            _dt=tf.transpose(delta_t,[1,0])
            _dt=tf.reshape( _dt, [ -1 ,1] )
            _X=tf.concat( 1, [_dt, _X ])#Put the time information in the first dimension
        _X = tf.split(0, self.n_steps, _X) # n_steps * (batch_size, self.n_hidden)

        #version 0.9
        outputs, states = tf.nn.rnn(cell, _X, initial_state=_istate, sequence_length=early_stop)
        #outputs, states = rnn.rnn(cell, _X, initial_state=_istate, sequence_length=early_stop)

        if isinstance(cell,rnn_cell.GRUCell):
            po=tf.pack(outputs)
            tf_outputs=tf.split(2,2,po)[0]
        else:
            tf_outputs=tf.pack(outputs)#(n_steps, batch_size, self.n_hidden)

        tf_outputs2d=tf.reshape(tf_outputs,[-1,self.n_hidden])
        #( n_steps x batch_size , self.n_hidden)  x   (self.n_hidden, self.n_classes)  + self.n_classes=
        #( n_steps x batch_size, self.n_classes)
        pred_y2d=tf.matmul( tf_outputs2d , _weights['out']) + _biases['out']
        return pred_y2d


    writer=None
    def write(self,fd=None):
        sess=self.session
        if self.writer is None:
            self.writer = tf.train.SummaryWriter(self.model_folder+'/summary',sess.graph)# graph_def depricated

        targets=[self.global_step,self.accuracy,self.accuracy_summary,self.bias_out_summary,self.cost,self.cost_summary]

        #Run sess.run only once so that only 1 thing dequeued
        step,acc,acc_str,b_out_str,loss, cost_str = sess.run(targets, feed_dict=fd)
        self.writer.add_summary( acc_str,step)# self.global_step)
        self.writer.add_summary( b_out_str, step)#self.global_step)
        self.writer.add_summary( cost_str,step)# self.global_step)
        self.writer.flush()
        return acc,loss
