'''

For each dataset, define a function that takes a records file and returns a batch

leave mutable args as params but hardcode dataset specific params

'''

import tensorflow as tf
import time

batch_size= 128
num_threads=4
capacity= 2000 #min(4*batch_size,1000)
n_steps=10
n_input=2308
n_context=15
num_epochs=1#1 for test data

example = tf.train.Example
feature = tf.train.Feature
features = lambda d: tf.train.Features(feature=d)
feature_list = lambda l: tf.train.FeatureList(feature=l)
feature_lists = lambda d: tf.train.FeatureLists(feature_list=d)
sequence_example = tf.train.SequenceExample

def _float_feature_array(value):
    return tf.train.Feature(float_list=tf.train.FloatList( value= value.astype('float')  ))
def _int64_feature_array(value):
    return tf.train.Feature(int64_list=tf.train.Int64List( value= value.astype('int')  ))
def _int64_feature(value):
    return tf.train.Feature( int64_list=tf.train.Int64List( value=[int(value)]))
def _bytes_feature(value):
    return tf.train.Feature( bytes_list=tf.train.BytesList( value=[float(value)]))


def todense(spT,shape=None):
    shape=shape or spT.shape
    return tf.sparse_to_dense(spT.indices,shape,spT.values)
def _flip_class(y):
    return tf.sign(y + tf.ones_like(y)) - tf.abs(y)
def _tf_stack( tensors):
    r=tf.rank(tensors[0])
    tensors_ = [tf.expand_dims(t,-1) for t in tensors]
    return tf.concat( r,tensors_)

##Dataset unstructured*
#unstructured_train has 808919 records
#unstructured_test has 201912 records

def unstructured22( batch_size=5,num_epochs=1,fname=None):
    if fname is None:
        fname='data/unstructured22.tfrecords'#default for debugging

    n_input=2308
    filename_queue = tf.train.string_input_producer([fname],num_epochs=num_epochs)
    reader  = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    parsed_features = tf.parse_single_example(
            serialized_example,
            features={
                    'name':tf.FixedLenFeature([],dtype=tf.int64),
                    'row':tf.FixedLenFeature([],dtype=tf.int64),
                    'data':tf.VarLenFeature(dtype=tf.float32),
                    'index':tf.VarLenFeature(dtype=tf.int64),
                    'y':tf.FixedLenFeature([],dtype=tf.int64),
                    },
            )

    batch=tf.train.batch_join([ parsed_features ],batch_size)
    batch['x']=tf.sparse_merge(sp_ids=batch['index'],sp_values= batch['data'] , vocab_size=n_input )
    batch['x']=todense( batch['x'])
    ##This returns batch['y'] as a 1d vector which I think will now be standard

    return batch#maybe also return reader and queue later


#def time_data_stream(object):
#	def __init__(self,stream,batch_size,num_threads,N=None,capacity=100):

#Useful function for timing different data reading methods
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



#Good for multiple files: #from parallel_reader import ParallelReader

class DataStream(object):
    def __init__(self, batch_fn, batch_size, fname=None, num_epochs=None):

        #batch_fn : function that produces a tensor representing a dequeued batch from a file
        #batch_size: number of examples per iter
        #num_epochs (optional): producer will throw OutOfRangeError after num_epochs work units created
        #           if not specified, method should rely on manually calling coord.request_stop()

        self.num_epochs=num_epochs
        self.batch_size=batch_size
        self.batch=batch_fn(num_epochs=self.num_epochs,batch_size=batch_size,fname=fname)

    def get_batch(self):
        print 'WARN:method only for debug'
        sess=tf.Session()
        sess.run(tf.initialize_all_variables() )
        tf.train.start_queue_runners(sess=sess)
        return sess.run(self.batch)


def default_data_stream(batch_size=128,num_epochs=1):
    tf.reset_default_graph()
    print 'Warning TFgraph reset!'
    #fname="/home/chris/models/train_data/train.tf.records"

    batch_fn=unstructured22 #default tfrecords for debuging
    stream=DataStream(batch_fn,num_epochs=num_epochs,batch_size=batch_size)
    #stream=SparseDataStream(n_steps=10,batch_size=batch_size,n_input=2308,n_context=15,filename=fname)
    return stream







