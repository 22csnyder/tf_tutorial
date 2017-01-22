import tensorflow as tf
import time

#from model_flags import *


def _int64_feature(value):
    return tf.train.Feature( int64_list=tf.train.Int64List( value=[value]))
def _int64_feature_array(value):
    return tf.train.Feature(int64_list=tf.train.Int64List( value= value.astype('int')  ))
def _bytes_feature(value):
    return tf.train.Feature( bytes_list=tf.train.BytesList( value=[value]))
def _float_feature(value):
    return tf.train.Feature( float_list=tf.train.FloatList(value=[float(value)]))
def _float_feature_array(value):
    return tf.train.Feature(float_list=tf.train.FloatList( value= value.astype('float')  ))
def _sparse_float_feature( indices, data):
    features=dict()
    for idx,value in zip(indices,data):
        features[str(idx)]=_float_feature( float( value) )
    return tf.train.Features( feature=features ) #expected tf.train.Feature
def _sparse_float_feature( indices, data):
    features=dict()
    for idx,value in zip(indices,data):
        features[str(idx)]=_float_feature( float( value) )
    return tf.train.Features( feature=features ) #expected tf.train.Feature
def _sparse_feature_list( csr ):
    return tf.train.FeatureList( feature=[_sparse_float_feature(x.indices,x.data) for x in csr])



#def time_data_stream(object):
#	def __init__(self,stream,batch_size,num_threads,N=None,capacity=100):
def time_data_stream(batch_size=128,num_threads=4,N=1000,capacity=100,stream=None):
	if stream is None:
		stream=default_data_stream()

	if N is None:#How is it going to stop?
		assert(stream.num_epochs is not None)
	else:
		var=tf.Variable( tf.constant(0,dtype=tf.int64) )
		limiter=tf.count_up_to(var,N)

	batch=stream.stream_batch(batch_size,num_threads,capacity)

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
		print 'batch_size',batch_size
		print 'n iter=',i
		print 'Time per batch:',(t2-t1)/i
		print 'Total Time:',t2-t1#,'\n\t',t1-t0
		coord.request_stop( e )
	finally:
		coord.request_stop()
		coord.join(threads, stop_grace_period_secs=100)
		sess.close()

###This is for a SINGLE example....	
def _flip_class(y):
    return tf.sign(y + tf.ones_like(y)) - tf.abs(y)
def _tf_stack( tensors):
    r=tf.rank(tensors[0])
    tensors_ = [tf.expand_dims(t,-1) for t in tensors]
    return tf.concat( r,tensors_)
def _get_early_stop(y):
    #r=tf.rank(y)
    return tf.reduce_sum(  tf.sign(tf.ones_like(y)+y))# , r-1 )

class DataStream(object):
    def read_and_decode_single_example(self):
        self.filename_queue = tf.train.string_input_producer([self.filename],num_epochs=self.num_epochs)
        #tf.add_to_collection('queues',self.filename_queue)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(self.filename_queue)
        p = tf.parse_single_example(serialized_example,features=self.tf_features)
        half_y=p['f_y']
        not_y= _flip_class(half_y)
        #_y = _tf_stack2D( [half_y, not_y])
        _y = _tf_stack([half_y, not_y])
        _y=tf.cast(_y, tf.float32)
        _x=tf.reshape( p['f_x'] , [self.n_steps,self.n_input] )
        es=_get_early_stop(half_y)
        _ma= tf.cast(p['f_ma'], tf.bool )
        #return [ _x , _y , p['f_ma'], p['f_pvec'], p['_t']]   
        return [ _x , _y , es, _ma, p['f_pvec'], p['_t']]


    def __init__(self,filename,n_steps=None,n_input=None,n_context=None,num_epochs=1,batch_size=128):
        self.filename=filename

        #flags = tf.app.flags
        #FLAGS = flags.FLAGS
        #self.model_dir=FLAGS.model_dir

        self.batch_size= batch_size
        self.num_threads=4
        self.capacity=2000# min(4*batch_size,1000)
        self.n_steps=10
        self.n_input=2308
        self.n_context=15
        self.num_epochs=num_epochs#1 for test data

        #self.batch_size= 128 or FLAGS.batch_size
        #self.num_threads=4 or FLAGS.num_threads
        #self.capacity= 400 or FLAGS.capacity
        #self.n_steps=10 or FLAGS.n_steps
        #self.n_input=2380 or FLAGS.n_input
        #self.n_context=17 or FLAGS.n_context
        #self.num_epochs=2 or FLAGS.num_epochs

        #self.num_epochs=num_epochs or FLAGS.num_epochs
        #self.n_steps=n_steps or FLAGS.n_steps
        #self.n_input=n_input or FLAGS.n_input
        #self.n_context=n_context or FLAGS.n_context

        size=self.n_steps * self.n_input
        #tf.reset_default_graph()
        self.tf_features={
        'f_x': tf.FixedLenFeature([size],tf.float32),
        'f_y': tf.FixedLenFeature([self.n_steps],tf.int64),
        'f_ma': tf.FixedLenFeature([self.n_steps],tf.int64),
        'f_pvec': tf.FixedLenFeature([self.n_context], tf.float32),
        '_t': tf.FixedLenFeature([self.n_steps],tf.float32)
        }
        self.inputs=self.read_and_decode_single_example()
        #self.batch= tf.train.batch( self.inputs, batch_size=self.batch_size,num_threads=self.num_threads,capacity=self.capacity)
        self.batch= tf.train.shuffle_batch( self.inputs, batch_size=self.batch_size,num_threads=self.num_threads,capacity=self.capacity,min_after_dequeue=500)

    def np_inputs(self):
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        threads=tf.train.start_queue_runners(sess=sess)
        inputs=sess.run( self.inputs )
        return inputs

def default_data_stream():
	tf.reset_default_graph()
	print 'Warning TFgraph reset!'
	fname="/home/chris/models/train_data/train.tf.records"
	stream=DataStream(n_steps=10,n_input=2308,n_context=15,filename=fname)
	return stream







