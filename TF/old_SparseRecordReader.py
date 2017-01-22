import tensorflow as tf
import time

#from model_flags import *

example = tf.train.Example
feature = tf.train.Feature
features = lambda d: tf.train.Features(feature=d)
feature_list = lambda l: tf.train.FeatureList(feature=l)
feature_lists = lambda d: tf.train.FeatureLists(feature_list=d)
sequence_example = tf.train.SequenceExample

#int64_feature = lambda v: feature(int64_list=tf.train.Int64List(value=v))
def _float_feature_array(value):
    return tf.train.Feature(float_list=tf.train.FloatList( value= value.astype('float')  ))
def _int64_feature_array(value):
    return tf.train.Feature(int64_list=tf.train.Int64List( value= value.astype('int')  ))
def _int64_feature(value):
    return tf.train.Feature( int64_list=tf.train.Int64List( value=[int(value)]))
def _bytes_feature(value):
    return tf.train.Feature( bytes_list=tf.train.BytesList( value=[float(value)]))


def fn_batch():
		filename='data/sparse.train.tfrecords'
		#flags = tf.app.flags
		#FLAGS = flags.FLAGS
		#model_dir=FLAGS.model_dir

		batch_size= 128
		num_threads=4
		capacity= 2000 #min(4*batch_size,1000)
		n_steps=10
		n_input=2308
		n_context=15
		num_epochs=1#1 for test data

		#size=n_steps * n_input
		#inputs=read_and_decode_single_example()#DEBUG
		##DEBUG BEGIN
		filename_queue = tf.train.string_input_producer([filename],num_epochs=num_epochs)
		reader  = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		#batch_serialized_examples = tf.train.batch([serialized_example], batch_size)
		#parsed_features = tf.parse_example(batch_serialized_examples,features= keys_to_features)
		context_features, sequence_features = tf.parse_single_sequence_example(
			serialized_example,
			context_features={
					'name':tf.FixedLenFeature([1],dtype=tf.int64),
					'y':tf.VarLenFeature(dtype=tf.int64),
					'mask':tf.VarLenFeature(dtype=tf.int64),
					'context':tf.FixedLenFeature([n_context],dtype=tf.float32),
					'time': tf.VarLenFeature(dtype=tf.float32),
					},
			sequence_features={
					'data':tf.VarLenFeature(dtype=tf.float32),
					'index':tf.VarLenFeature(dtype=tf.float32),
					}
			)
		features={}
		features.update(context_features)
		features.update(sequence_features)
		#_batch=tf.train.batch_join([ features ],batch_size, dynamic_pad=True)
		_batch=tf.train.batch_join([ features ],batch_size)
		x=tf.sparse_merge(sp_ids=_batch['index'],sp_values= _batch['data'] , vocab_size=n_input )
		batch={}
		batch['x']=x
		batch['y']=todense(_batch['y'])
		batch['mask']=todense(_batch['mask'])
		batch['context']= _batch['context']
		batch['time']=todense(_batch['time'])

		sess=tf.Session()
		sess.run(tf.initialize_all_variables() )
		tf.train.start_queue_runners(sess=sess)

		return sess.run(batch)



#def time_data_stream(object):
#	def __init__(self,stream,batch_size,num_threads,N=None,capacity=100):
def time_data_stream(batch_size=128,num_threads=4,N=1000,capacity=100,stream=None):
	if stream is None:
		stream=default_data_stream(batch_size)

	if N is None:#How is it going to stop?
		assert(stream.num_epochs is not None)
	else:
		var=tf.Variable( tf.constant(0,dtype=tf.int64) )
		limiter=tf.count_up_to(var,N)

	#batch=stream.stream_batch(batch_size,num_threads,capacity)
        batch=stream.batch

	init=tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)
	coord = tf.train.Coordinator()
	threads=tf.train.start_queue_runners(coord=coord,sess=sess)
	#threads=tf.train.start_queue_runners(coord=coord,sess=sess)
	t0=time.time()
	i=0
	try:
		while not coord.should_stop():
#			if N is not None:
#				if i>=N:
#					coord.request_stop()
				#tf.count_up_to(var,N)
			if N is not None:
				sess.run(limiter)
			if i==0:
				t1=time.time()
			if i%10==0:
				print 'i=',i
			i=i+1
			b=sess.run(batch)
	#except tf.errors.OutOfRangeError as e:
	#except tf.errors as e:
	except Exception, e:
		t2=time.time()
		print 'num_threads',num_threads
                print 'actual_numthreads',len(threads)
		print 'batch_size',stream.batch_size
		print 'n iter=',i
		print 'Time per batch:',(t2-t1)/i
		print 'Total Time:',t2-t1#,'\n\t',t1-t0
		coord.request_stop( e )
	finally:
		coord.request_stop()
		coord.join(threads, stop_grace_period_secs=100)
		sess.close()

###This is for a SINGLE example....	
#def _flip_class(y):
#    return tf.sign(y + tf.ones_like(y)) - tf.abs(y)
def _tf_stack( tensors):
    r=tf.rank(tensors[0])
    tensors_ = [tf.expand_dims(t,-1) for t in tensors]
    return tf.concat( r,tensors_)

def todense(spT,shape=None):
    shape=shape or spT.shape
    return tf.sparse_to_dense(spT.indices,shape,spT.values)

class SparseDataStream(object):
	#TODO: we have the name of each pt stored but not using it
	def read_and_decode_single_example(self):
		filename_queue = tf.train.string_input_producer([self.filename],num_epochs=self.num_epochs)
		reader  = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		#batch_serialized_examples = tf.train.batch([serialized_example], batch_size)
		#parsed_features = tf.parse_example(batch_serialized_examples,features= keys_to_features)

		context_features, sequence_features = tf.parse_single_sequence_example(
				serialized_example,
				context_features={
						'name':tf.FixedLenFeature([1],dtype=tf.int64),
						'y':tf.VarLenFeature(dtype=tf.int64),
						'mask':tf.VarLenFeature(dtype=tf.int64),
						'context':tf.FixedLenFeature([self.n_context],dtype=tf.float32),
						'time': tf.VarLenFeature(dtype=tf.float32),
						},
				sequence_features={
						'data':tf.VarLenFeature(dtype=tf.float32),
						'index':tf.VarLenFeature(dtype=tf.float32),
						}
				)
		return sequence_features

		return [sequence_features['index'],sequence_features['data']]#DEBUG
		features={}
		features.update(context_features)
		features.update(sequence_features)
		mask=todense(features['mask'], [self.n_steps])
		features['es']=tf.reduce_sum(mask)
		y=todense(features['y'],[self.n_steps])
		features['y']= tf.cast( _tf_stack([ mask-y, y]), tf.float32 )
		features['mask']=tf.cast( mask, tf.bool)
		features['time']=todense(features['time'],[self.n_steps])

		return [features[key] for key in ['index','data','y','es','mask','context','time']]

	def __init__(self,filename,n_steps=None,n_input=None,n_context=None,num_epochs=1,batch_size=128):
		self.filename=filename
		#flags = tf.app.flags
		#FLAGS = flags.FLAGS
		#self.model_dir=FLAGS.model_dir

		self.batch_size= batch_size
		self.num_threads=4
		self.capacity= 2000 #min(4*batch_size,1000)
		self.n_steps=10
		self.n_input=2308
		self.n_context=15
		self.num_epochs=num_epochs#1 for test data

		#size=self.n_steps * self.n_input
		#self.inputs=self.read_and_decode_single_example()#DEBUG
		##DEBUG BEGIN
		filename_queue = tf.train.string_input_producer([self.filename],num_epochs=self.num_epochs)
		reader  = tf.TFRecordReader()
		_, serialized_example = reader.read(filename_queue)
		#batch_serialized_examples = tf.train.batch([serialized_example], batch_size)
		#parsed_features = tf.parse_example(batch_serialized_examples,features= keys_to_features)
		context_features, sequence_features = tf.parse_single_sequence_example(
			serialized_example,
			context_features={
					'name':tf.FixedLenFeature([1],dtype=tf.int64),
					'y':tf.VarLenFeature(dtype=tf.int64),
					'mask':tf.VarLenFeature(dtype=tf.int64),
					'context':tf.FixedLenFeature([n_context],dtype=tf.float32),
					'time': tf.VarLenFeature(dtype=tf.float32),
					},
			sequence_features={
					'data':tf.VarLenFeature(dtype=tf.float32),
					'index':tf.VarLenFeature(dtype=tf.float32),
					}
			)
		features={}
		features.update(context_features)
		features.update(sequence_features)
		#_batch=tf.train.batch_join([ features ],self.batch_size, dynamic_pad=True)
		_batch=tf.train.batch_join([ features ],self.batch_size)
		x=tf.sparse_merge(sp_ids=_batch['index'],sp_values= _batch['data'] , vocab_size=self.n_input )
		batch={}
		batch['x']=x
		batch['y']=todense(_batch['y'])
		batch['mask']=todense(_batch['mask'])
		batch['context']= _batch['context']
		batch['time']=todense(_batch['time'])
		self.batch=batch


		#DEBUG
		##shapes=[ [self.batch_size, self.n_steps, self.n_input
		#batch=tf.train.batch_join([ self.inputs ], batch_size=self.batch_size, capacity=self.capacity,dynamic_pad=False)
		##self.batch=batch

		##batch=tf.train.batch([ self.inputs ], batch_size=self.batch_size, num_threads=self.num_threads, capacity=self.capacity)
		##batch=tf.train.batch_join([ self.inputs ], batch_size=self.batch_size, capacity=self.capacity,dynamic_pad=True)
		##batch=tf.train.batch_join( self.inputs , batch_size=self.batch_size, capacity=self.capacity,dynamic_pad=True)
		##batch=tf.train.shuffle_batch_join([ self.inputs ], batch_size=self.batch_size, num_threads=self.num_threads, capacity=self.capacity,min_after_dequeue=500)
		##batch=tf.train.shuffle_batch_join([ self.inputs ], batch_size=self.batch_size, capacity=self.capacity,min_after_dequeue=500)

		##DEBUG
		#index=batch['index']
		#data=batch['data']
		##index,data=batch[:2]
		#x=tf.sparse_merge(sp_ids=index,sp_values=data, vocab_size=n_input)
		#self.batch=x
		###self.batch=[batch[key] for key in ['x','y','es','mask','context','time']]
		##self.batch= [ todense(x) ] + batch[2:]
	def get_batch(self):
		sess=tf.Session()
		sess.run(tf.initialize_all_variables() )
		tf.train.start_queue_runners(sess=sess)
		return sess.run(self.batch)

	def np_inputs(self):
		init = tf.initialize_all_variables()
		sess = tf.Session()
		sess.run(init)
		threads=tf.train.start_queue_runners(sess=sess)
		inputs=sess.run( self.inputs )
		return inputs

def default_data_stream(batch_size=128):
    tf.reset_default_graph()
    print 'Warning TFgraph reset!'
    #fname="/home/chris/models/train_data/train.tf.records"
    fname="data/sparse.train.tfrecords"
    stream=SparseDataStream(n_steps=10,batch_size=batch_size,n_input=2308,n_context=15,filename=fname)
    return stream







