import tensorflow as tf
import time
import threading
import os
from tensorflow.python.ops import variable_scope as vs
from sklearn.cross_validation import train_test_split
import numpy as np
from utils import chunks, tf_counter, make_folders
from sklearn.metrics import roc_curve, auc

#from tf.contrib.layers import summarize_tensor
from TF.utils import Accumulator
from contextlib import contextmanager

class Basic_Graph(object):
    n_classes=2#default

    _prefix=None
    @property
    def prefix(self):
        if self.is_training:
            return 'train'

        elif self._prefix is not None:
            return self._prefix

        else:
            return 'test'

    @property
    def metrics(self):
        return self.prefix+'_'+'metrics'

    @property
    def update_ops(self):
        return self.prefix+'_'+'update_ops'

    @property
    def summaries(self):
        return self.prefix+'_'+'summaries'

    def __init__(self):
        raise NotImplementedError
    ######      An example init method taken from nn_model.py    #######
    ######      The critical part is calling _inference_graph(), _analysis_graph() ####
    ######        at the end if you want that functionality    ####
    #def __init__(self, n_features,learning_rate=0.01, n_hidden=50, is_training=False):
        #    self.is_training=is_training
        #    self.learning_rate=learning_rate
        #    self.n_features=n_features
        #    self.n_hidden=n_hidden

    #    self.x =tf.placeholder("float", [None, self.n_features], name="x")
    #    self.y =tf.placeholder(tf.int64,  name="y")

    #    #Define graph
    #    self._inference_graph()
    #    self._analysis_graph()
    #    if self.is_training:
        #        self._loss_graph()
        #	self._summaries_graph()

    def _inference_graph(self):
        raise NotImplementedError
    #     ******  """          A sample to follow:    """   *******
    #x1=_simple_tanh_layer( self.x , output_size=self.n_hidden , scope='embedding')
    #x2=_simple_tanh_layer( x1 ,scope='layer1')
    #x3=_simple_tanh_layer( x2 ,scope='layer2')
    #self.logits = _linear( x3 ,output_size=2, scope='projection')
    #self.class_probs=tf.nn.softmax( self.logits )
    #self.y_hat= tf.cast( tf.argmax( self.class_probs, 1), tf.float32 )

    def _analysis_graph(self):
        #Default analysis graph behavior. This should be called in your init method
        #You can override this method if that suites you.

        #To use this method self.y and self.y_hat have to be defined#

        ##y should be passed in as a 1d matrix with (1..k classes)
        #self.y1d=tf.squeeze(tf.split( split_dim=1, num_split=2, value=self.y )[1])
        #self.is_correct= tf.equal( tf.cast(self.y_hat,tf.int64), tf.cast(self.y1d,tf.int64) )

        #y should already be int64
        self.is_correct= tf.equal( tf.cast(self.y_hat,tf.int64), self.y )
        self.num_correct= tf.reduce_sum( tf.cast( self.is_correct, tf.float32) )
        self.num_possible=tf.size(self.is_correct) #y1d
        self.accuracy=tf.div( tf.cast(self.num_correct,tf.float32), tf.cast(self.num_possible,tf.float32) )

    def _loss_graph(self):
        #  Default loss graph.. should work for many graphs
        #      to make use of this graph just make sure you define "self.logits"
        #      which should be the "unsquashed" output of the network
        #      
        #      if you want to use your own loss graph, that's fine. Just make sure it
        #      defines an operation called "self.train_op"

        #To use this method as is, self.logits has to be defined
        self.y_hot=tf.one_hot( self.y,  self.n_classes)
        float_y=tf.cast(self.y_hot,tf.float32)
        self.cross_entropy=tf.nn.softmax_cross_entropy_with_logits(self.logits, float_y, name='cross_entropy') # (n_steps x batch_size,) 
        self.loss=tf.reduce_mean(self.cross_entropy, name='loss')
        ## Optional regularization that will occur if defined self.lambda ##
        if hasattr(self,'TF_lambdaL2'):
            if self.TF_lambdaL2 is not None:
                weights= tf.get_collection(tf.GraphKeys.WEIGHTS)
                #l2_reg= tf.reduce_sum(tf.pack([tf.l2_loss(w) for w in weights]))
                l2_reg=tf.add_n([tf.nn.l2_loss(w) for w in weights])
                self.loss+= self.TF_lambdaL2* l2_reg


        if self.is_training:
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            #self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        #Spelled out version of the below:
            #self.opt= tf.train.AdamOptimizer(self.learning_rate)
            #self.gradients=self.opt.compute_gradients(self.loss)
            #self.train_op=self.opt.apply_gradients(self.gradients)

    def summary_mean(self, name, tensor):
        stream,update = tf.contrib.metrics.streaming_mean(
            tensor,
            metrics_collections=self.metrics,
            updates_collections=self.update_ops)
        tf.summary.scalar(self.prefix+'_'+ name,stream,[self.summaries])


    def _summaries_graph(self):
        '''create summaries for graph'''
        ####If these things are defined somewhere in your graph this could be useful####
        #prefix=''
        col=[self.summaries]
        #    prefix+='train'
        #if self.is_training:#NOTE - do we need this part
        if hasattr(self, 'loss'):
            tf.summary.scalar('batch_loss', self.loss,col)
            stream_loss,update_loss= tf.contrib.metrics.streaming_mean(
                self.loss,
                metrics_collections=self.metrics,
                updates_collections=self.update_ops)
            tf.summary.scalar(self.prefix+'_'+'loss',stream_loss,col)

        if hasattr(self, 'accuracy'):
            tf.summary.scalar('batch_accuracy', self.accuracy,col)
            stream_acc,update_acc= tf.contrib.metrics.streaming_accuracy(
                predictions=self.y_hat,labels=self.y)
            tf.add_to_collection(self.update_ops,update_acc)
            tf.add_to_collection(self.metrics,stream_acc)
            tf.summary.scalar(self.prefix+'_'+'acc',stream_acc,col)

        if hasattr(self, 'logits'):
            tf.summary.histogram('batch_logits', self.logits,col)

        if hasattr(self,'class_probs'):
            tf.summary.histogram('class_probs', self.class_probs,col)

        if hasattr(self,'y_hat') and self.n_classes == 2:
            self.bool_y=tf.cast(self.y, tf.float32)
            self.bool_y_hat=tf.cast(self.y_hat, tf.float32)
            stream_auc,update_auc= tf.contrib.metrics.streaming_auc(
                predictions=self.bool_y_hat,labels=self.bool_y,
                metrics_collections=self.metrics,
                updates_collections=self.update_ops)
            tf.summary.scalar(self.prefix+'_'+'auc',stream_auc,col)

        #print 'this occurs:,'col[0]

        #(attr,reduce_method,tag)
        #self.cumulative_summaries=[]
        #self.cumulative_summaries.append( dict(attr='loss',tag='meanloss')
        #self.cumulative_summaries.append( dict(attr='loss',reduction='max',tag='maxloss')
        #self.log_names=list(set([s[0] for s in self.cumulative_summaries if hasattr(self,s[0])]))
        #self.log_tensors=[getattr(self,name) for name in self.log_names)]

            #attr='loss'
            #if hasattr(self,attr):
                #    name=prefix+='_'+attr
                #    summarize_tensor(tensor=getattr(self,attr),tag=name)


            #n_classes=self.y.get_shape().as_list()[1]
            #batch_mean=tf.reduce_mean( self.y1d )


#def summ_dict():
    #returns a summary dict with default values
    #    yield dict(reduction='mean',tag=None,type='scalar')

from utils import ModelKeeper

#@contextmanager
#def SummaryCollection(collection_key):


class Basic_Model(object):
    _default_batch_size=256
    def initialize(self):
        init=tf.global_variables_initializer()
        self.session.run(init)
        self.session.run(tf.local_variables_initializer())
        self.threads=tf.train.start_queue_runners(coord=self.coord,sess=self.session)

    def close(self):
        coord.request_stop()
        coord.join(threads)
        self.session.close()

    def load_model(self,model=None):
        if model is None:
            make=True
            model=self.scope
        elif model is not None:
            make=False
            checkpoint_folder=ModelKeeper.get_folders(model,make)
            model_folder= os.path.join(self.model_dir,model)
            checkpoint_folder=os.path.join(self.model_folder, 'checkpoints')

        if not hasattr(self,'saver'):
            self.saver=tf.train.Saver(max_to_keep=None)

        #ckpt = tf.train.get_checkpoint_state(self.model_folder)
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_folder)

        if ckpt:
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            print('Loading model: %s'%ckpt.model_checkpoint_path)

    def save(self):
        if not hasattr(self,'saver'):
            self.saver=tf.train.Saver()

        f=os.path.join(self.checkpoint_folder,'ckpts')
        save_path = self.saver.save(self.session, f, global_step=self.epoch)
        #save_path = self.saver.save(self.session, f, global_step=self.step)
        print("Model saved in file: %s" % save_path)

    def __init__(self, graph, graph_kwargs, session=None, scope='Basic_Model', train_inputs=None, test_inputs=None, n_train_examples=None):

        self.n_train_examples=n_train_examples

        self.graph=graph
        self.scope=scope
        self.session=session or tf.Session()

        self.train_inputs=None
        self.test_inputs =None
        ##Is it a fully connected model that reads from a TFRecords file?
        if train_inputs:
            self.train_stream=train_inputs
            self.train_inputs=train_inputs.batch

        if test_inputs:
            self.test_stream=test_inputs
            self.test_inputs=test_inputs.batch

        self.model_dir,self.model_folder,self.checkpoint_folder,self.summary_folder=ModelKeeper.get_folders(scope,make=True)
        #self.cnt_epoch,self.cnt_epoch_increment= tf_counter('counter_train_epochs')
        self.cnt_epoch,self.cnt_epoch_add_one= tf_counter('counter_train_epochs')
        self.cnt_step,self.cnt_step_add_one= tf_counter('counter_train_steps')

        self._define_graphs(**graph_kwargs)

        #Turn off for the moment#DEBUG
        #with tf.control_dependencies( [self.cnt_step_add_one] ):
        #    self.train_op=tf.group( self.train_graph.train_op )
        self.train_op=self.train_graph.train_op

        self.coord = tf.train.Coordinator()

        #writer
        self.writer = tf.summary.FileWriter(self.summary_folder ,self.session.graph)# graph_def depricated
        self.merged=tf.summary.merge_all()

        #self.update_op= tf.group( tf.get_collection('update_ops'))

        with tf.device('/cpu:0'):
            #saver has to be constructed after the whole graph is built
            self.save_vars=[v for v in tf.global_variables() if not 'input_producer' in v.name]
                            #() if not 'input_producer' in v.name]
            self.saver=tf.train.Saver(var_list=self.save_vars,
                                      keep_checkpoint_every_n_hours=2 )


    def _define_graphs(self,**kwargs):
        with tf.variable_scope(self.scope,reuse=None):
            kwargs['is_training']=True
            if self.train_inputs:
                kwargs['connected_inputs']=self.train_inputs
            self.train_graph= self.graph( **kwargs )

        with tf.variable_scope(self.scope,reuse=True):
            kwargs['is_training']=False
            if self.test_inputs:
                kwargs['connected_inputs']=self.test_inputs
            #could define prefix here
            self.test_graph= self.graph(**kwargs )


        #Switches to use to change fit/eval/predict usage
        self.default_train_graph= self.train_graph
        self.default_test_graph=  self.test_graph

    def write(self,summary_list):
        for s in summary_list:
            #Up to you if you want to use epoch vs step
            self.writer.add_summary( s ,self.epoch )
            self.writer.flush()

    @property
    def step(self):
        #self.cnt_step is a tf.Tensor so we have to access it like this
        return self.session.run( self.cnt_step )


    @property
    def epoch(self):
        return self.session.run( self.cnt_epoch )
    #def epoch(self):
        #    assert(self.n_train_examples is not None)
        #    return np.round( float(self.step) / self.n_train_examples
        #    #return self.session.run( self.cnt_epoch )

    #@property
    #def fetches(self):
    #    fetches=[]
    #    fetches.extend(  tf.get_collection('update_ops') )

    #@contextmanager
    #def process(self,graph,inputs=None):

    def feed_fn(self,graph,input_x=None, input_y=None,n_iter=None,batch_size=None):
        #TODO: augment this to take larger than memory data
        #graph is the computataion graph (train,test)
        assert( n_iter or input_x is not None), 'either n_iter or inputs must be given'
        assert( not n_iter or input_x is None), 'both n_iter and inputs can not be given'

        #fully connected model:
        if input_x is None:
            for _ in xrange(n_iter):
                yield None
            return

        chunk_size=batch_size or self._default_batch_size
        if input_y is None:
            #Would be easier with "yield from" (python3 only)
            x_=iter(chunks(input_x,chunk_size))
            while True:
                x_batch = next(x_)
                feed_dict = {graph.x:x_batch}
                yield feed_dict
            return
        if input_x is not None and input_y is not None:
            iter_data=iter( chunks([input_x,input_y],chunk_size,axis=1) )
            while True:
                x_batch,y_batch=next(iter_data)
                #x_batch,y_batch=next(x_), next(y_)
                feed_dict = {graph.x: x_batch, graph.y: y_batch}
                yield feed_dict
            return
        else:
            raise NotImplementedError('shouldnt happen')

    def data_loop(self,graph,inputs=None,n_iter=None,batch_size=None):
        assert( n_iter or inputs is not None), 'either n_iter or inputs must be given'
        assert( not n_iter or inputs is None), 'both n_iter and inputs can not be given'

        if inputs is None:
            input_x = None
            input_y = None
        elif isinstance(inputs,dict):
            if 'x' in inputs.keys():
                input_x=inputs['x']
            if 'y' in inputs.keys():
                input_y=inputs['y']
            else:
                input_y=None


        fd_stream=self.feed_fn(graph, input_x, input_y, n_iter=n_iter,batch_size=batch_size)
        #print 'data_loop running.. '
        for feed_dict in fd_stream:
            with self.coord.stop_on_exception():
                yield feed_dict
                #if i>10000000:
                    #    raise Exception('inf loop:probably shouldnt happen?')

    def predict(self,inputs=None):
        g=self.default_test_graph

        self.reset_streaming_metrics()
        updates=tf.get_collection(g.update_ops)
        summaries=tf.get_collection(g.summaries)

        print 'entering predict loop'
        Y=[]; P=[]
        for feed_dict in self.data_loop(graph=g,inputs=inputs):
            #print 'pred loop'
            y_hat,class_prob,_,summ=self.session.run([g.y_hat,g.class_probs,updates,summaries], feed_dict=feed_dict)
            #print 'y_hat:',y_hat
            Y.append(y_hat)
            P.append(class_prob)

        print 'writing'

        if summ is not None:
            self.write(summ)
        print 'done writing'

        return np.concatenate(Y),np.concatenate(P)

    def evaluate(self,inputs=None):
        print "Evaluating data..."
        g=self.default_test_graph

        testX,testY=inputs or [None,None]
        y_hat,y_prob=self.predict(inputs)

        #test_acc=float(np.sum(np.equal(y_hat,testY)))/np.size(testY)
        #print 'Test Accuracy: ', test_acc

        #print 'testYshape:',testY.shape
        #print 'y_prob.shape:',y_prob.shape

        #assert(g.n_classes==2)#roc only defined for 2 classes
        #y_score=y_prob[:,1]

        #false_positive_rate, true_positive_rate, thresholds = roc_curve(testY, y_score)#actual,predictions
        #roc_auc = auc(false_positive_rate, true_positive_rate)
        #print 'AUC: ', roc_auc

        #eval_dict=dict(auc=roc_auc,
        #               #tpr=np.median(true_positive_rate),
        #               #fpr=np.median(false_positive_rate),
        #               acc=test_acc)
        #eval_dict['E[y_hat]']=np.mean(y_hat)
        #eval_dict['E[y_score]']=np.mean(y_score)
        #eval_dict['E[y_true]']=np.mean(testY)
        #summ = tf.Summary(value=[tf.Summary.Value(tag='test_'+key,simple_value=float(value)) for key,value in eval_dict.iteritems()])

        #self.write([summ])

        #return eval_dict

    def reset_streaming_metrics(self):
        #DEBUG -- does this reset queuerunner counter?
        self.session.run( tf.local_variables_initializer())

    #n_epochs must be specified not n_iter for the moment
    #def fit(self,inputs=None,n_epochs=1,n_iter=None,batch_size=None,
    def fit(self,inputs=None,n_epochs=None,batch_size=None,
            epoch_per_summary=1, epoch_per_checkpoint=5):
            #epoch_per_summary=5, epoch_per_checkpoint=25):
        #batch_size is not to be given if inputs is not given.
        #if fully connected, batch_size was baked in during graph construction
        print 'fitting..'

        #self.reset_streaming_metrics()#I think I should move this down to per epoch
        g=self.default_train_graph

        #logger=Accumulator(['train_acc','train_loss','train_E[y_hat]','train_E[y_score]'])

        t1=time.time()
        i=0

        #nag: n_iter is actually rounds short if batch_size doesn't line up
        if inputs is None:
            n_iter=self.train_stream.records_per_epoch/self.train_stream.batch_size#iter per epoch
        else:
            #n_iter=len(inputs['x'])
            n_iter=None

        for epoch in range(n_epochs):
            self.reset_streaming_metrics()
            if inputs is not None:#shuffle
                shuffle_ind = np.mgrid[0:len(inputs['x'])]
                np.random.shuffle( shuffle_ind )#inplace operation
                for key in inputs.keys():
                    inputs[key]= inputs[key][shuffle_ind]

            print 'running epoch',self.epoch+1,'...'

            for feed_dict in self.data_loop(graph=g,inputs=inputs,n_iter=n_iter,batch_size=batch_size):
                i+=1

                updates=tf.get_collection(g.update_ops)
                summaries=tf.get_collection(g.summaries)
                #print 'dtrain'
                #summ,_,_= self.session.run([summaries,g.d_train,updates], feed_dict=feed_dict)
                #print 'gtrain'
                #summ,_,_= self.session.run([summaries,g.g_train,updates], feed_dict=feed_dict)
                #print 'both train'
                summ,_,_= self.session.run([summaries,self.train_op,updates], feed_dict=feed_dict)
                #print 'worked'

            if self.epoch % epoch_per_summary == 0:
                if summ is not None:
                    self.write(summ)

            if self.epoch % epoch_per_checkpoint == 0:
                self.save()

                #logger.log([acc, loss, np.mean(y_hat),np.mean(y_score)], weight=y_hat.shape[0])#has to be correct order

            self.session.run(self.cnt_epoch_add_one)

        print 'threading.active_threads',threading.active_count()
        print 'n iter=',i
        t2=time.time()
        if i > 0:
            print 'Time per fit Step:',(t2-t1)/i
            if n_iter is not None:
                print 'Total time to fit per epoch:',(t2-t1)/n_epochs
            else:
                print 'Total time to fit',(t2-t1)

            #eval_dict=logger.reduce_mean()
            #train_summ = tf.Summary(value=[tf.Summary.Value(tag=key,simple_value=value) for key,value in eval_dict.iteritems()])


        else:
            print 'oops, process loop never occured. i=',i


