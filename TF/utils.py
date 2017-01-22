import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops,init_ops
import numpy as np
import os

##sometimes when stddev was too big I would get nan after exp
truncated_normal= init_ops.truncated_normal_initializer(mean=0.0,stddev=0.01,dtype=tf.float32)

random_normal= init_ops.random_normal_initializer(mean=0.0,stddev=1.0,dtype=tf.float32)

xavier=tf.contrib.layers.xavier_initializer(uniform=True,dtype=tf.float32)


##switch##
default_initializer= truncated_normal
##switch##

#Right multiply, since batch dim usually on left 
def _linear( inputs, output_size=None, scope=None,bias=True,reg=False):
	input_size=inputs.get_shape().as_list()[-1]
	output_size = output_size or input_size

	with vs.variable_scope(scope or "Linear"):
		W = vs.get_variable("Matrix", [input_size,output_size],
			initializer=default_initializer)
			#initializer=init_ops.random_normal_initializer(mean=0.0, stddev=1.0))
		if reg:
			tf.add_to_collection( tf.GraphKeys.WEIGHTS, W )

		output=tf.matmul(inputs,W)

		if bias:
			b=vs.get_variable('Bias', [1,output_size],
				initializer=default_initializer)
				#initializer=init_ops.random_normal_initializer(mean=0.0, stddev=1.0))
			#output+=b
			output=tf.add( output, b)

	return output


def _simple_layer(inputs,output_size=None,scope=None,summary=False,bias=True,squash=None,reg=False):
    if squash is None:
        raise ValueError('squash must be specified')
    if squash=='tanh':
        nonlinearity=tf.tanh
    elif squash=='relu':
        nonlinearity=tf.nn.relu
    elif squash=='relu6':
        nonlinearity=tf.nn.relu6
    elif squash=='sigm':
        nonlinearity=tf.sigmoid
    elif squash=='softplus':
        nonlinearity=tf.nn.softplus
        #nonlinearity=tf.sigm
    scope= scope or (squash + "_NN_layer")
    enc= _linear(inputs, output_size=output_size, scope=scope,bias=bias,reg=reg)
    #enc= _linear(**kwargs)
    activation=nonlinearity(enc,name=scope+'_activation')
    if summary:
        tf.contrib.layers.summarize_activation( activation )
    return activation

from functools import partial
_simple_relu_layer=partial(_simple_layer, squash='relu')
_simple_relu6_layer=partial(_simple_layer, squash='relu')
_simple_tanh_layer=partial(_simple_layer, squash='tanh')
_simple_sigm_layer=partial(_simple_layer, squash='sigm')
_simple_softplus_layer=partial(_simple_layer, squash='softplus')

def chunks(l, chunk_size,axis=0):
    """Yield successive n-sized chunks from l.
	If axis=1, then l is interpreted as a collection of items to chunk
        In this case, all of the items in l must have same length"""

    n=chunk_size
    if axis==0:
        for i in xrange(0, len(l), n):
            #Takes care of the tail end when len(l)%n ~= 0
            n=n- max(0, (i+n) - len(l) )
            yield l[i:i+n]#caution: generator not thread safe

    #Fancy way of saying input is list of iterables to be chunked
    elif axis==1:
        for i in xrange(0, len(l[0]), n):
            out=[]
            for item in l:
                n=n- max(0, (i+n) - len(item) )
                out.append( item[i:i+n] )
            yield out
    else: #Theoretically you could support higher axis numbers
        raise NotImplementedError

def make_folder(f):
    if not os.path.isdir(f):
        assert(not os.path.exists(f))#don't overwrite if is file
        os.mkdir(f)
def make_folders(fs):#order of folder creation important
    for f in fs:
        make_folder(f)

def tf_counter(name):
    counter=tf.Variable(0, name=name, trainable=False)
    counter_add_one=tf.assign(counter, counter+1 )
    return counter, counter_add_one


class ModelKeeper(object):
    @staticmethod
    def get_folders(model,make=False):
        model_dir=os.path.join( os.getcwd(), 'models' )
        #display_step = FLAGS.display_step
        model_folder= os.path.join(model_dir,model)
        if make is False:
            if not os.path.exists(model_folder):
                raise ValueError('Model does not already exist: %s'%model_folder)
        checkpoint_folder=os.path.join(model_folder, 'checkpoints')
        summary_folder=os.path.join(model_folder, 'summary')
        if make:
            make_folders([ model_dir, model_folder, checkpoint_folder, summary_folder] )
        return [ model_dir, model_folder, checkpoint_folder, summary_folder]

from collections import OrderedDict
class Accumulator(object):
    def __init__(self,ordered_var_names=None):
        self._weights=[]
        self._storage=OrderedDict()
        if ordered_var_names:
            self.keep_track(ordered_var_names)

    def keep_track(self, ordered_var_names):
        for name in ordered_var_names:
            self._storage[name]=[]

    def log(self, data_list,weight):
        #Assumed to be in same order
        if not len(data_list)==len(self._storage.keys()):
            raise ValueError('Must have one datum of each type')

        self._weights.append( weight )
        for key,datum in zip(self._storage.keys(), data_list):
            self._storage[key].append(datum)

    def reduce_mean(self):
        mean=dict()
        np_weights=np.array(self._weights)
        for key in self._storage.keys():
            mean[key]= np.average( self._storage[key], weights=np_weights).astype('float')
        return mean




















