import tensorflow as tf
import time
import os
from tensorflow.python.ops import variable_scope as vs
from sklearn.cross_validation import train_test_split
import numpy as np
from utils import chunks, tf_counter, make_folders
from sklearn.metrics import roc_curve, auc

#from tf.contrib.layers import summarize_tensor
from TF.utils import Accumulator


class Basic_Graph(object):
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
    #    self.y =tf.placeholder("float", [None, 2], name="y")

    #    #Define graph
    #    self._inference_graph()
    #    self._analysis_graph()
    #    if self.is_training:
    #        self._loss_graph()

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
        self.y1d=tf.squeeze(tf.split( split_dim=1, num_split=2, value=self.y )[1])
        self.is_correct= tf.equal( tf.cast(self.y_hat,tf.int64), tf.cast(self.y1d,tf.int64) )
        self.num_correct= tf.reduce_sum( tf.cast( self.is_correct, tf.float32) )
        self.num_possible=tf.size(self.y1d)
        self.accuracy=tf.div( tf.cast(self.num_correct,tf.float32), tf.cast(self.num_possible,tf.float32) )

    def _loss_graph(self):
        #  Default loss graph.. should work for many graphs
        #      to make use of this graph just make sure you define "self.logits"
        #      which should be the "unsquashed" output of the network
        #      
        #      if you want to use your own loss graph, that's fine. Just make sure it
        #      defines an operation called "self.train_op"

		#To use this method as is, self.logits has to be defined
        self.cross_entropy=tf.nn.softmax_cross_entropy_with_logits(self.logits, self.y, name='cross_entropy') # (n_steps x batch_size,) 
        self.loss=tf.reduce_mean(self.cross_entropy, name='loss')
        ## Optional regularization that will occur if defined self.lambda ##
        if hasattr(self,'TF_lambdaL2'):
            weights= tf.get_collection(tf.GraphKeys.WEIGHTS)
            #l2_reg= tf.reduce_sum(tf.pack([tf.l2_loss(w) for w in weights]))
            l2_reg=tf.add_n([tf.nn.l2_loss(w) for w in weights])
            self.loss+= self.TF_lambdaL2* l2_reg

        #Spelled out version of the below:
        self.opt= tf.train.AdamOptimizer(self.learning_rate)
        self.gradients=self.opt.compute_gradients(self.loss)
        self.train_op=self.opt.apply_gradients(self.gradients)

        #self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        #self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

    def _summaries_graph(self):
        ####If these things are defined somewhere in your graph this could be useful####
        #prefix=''
        #    prefix+='train'
        if self.is_training:
            if hasattr(self,'loss'):
                tf.scalar_summary('batch_loss',self.loss)
            if hasattr(self,'accuracy'):
                tf.scalar_summary('batch_accuracy',self.accuracy)
            if hasattr(self,'logits'):
                tf.histogram_summary('batch_logits',self.logits)
            if hasattr(self,'class_probs'):
                tf.histogram_summary('class_probs',self.class_probs)

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

class Basic_Model(object):
    def init_variables(self):
        self.session.run(tf.initialize_all_variables())
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
                self.saver=tf.train.Saver()
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
        print("Model saved in file: %s" % save_path)

    def __init__(self, graph,session=None,scope='Basic_Model',train_inputs=None, test_inputs=None, *args, **kwargs):
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

        ##'models' added to .gitignore
        #self.model_dir=os.path.join( os.getcwd(), 'models' )
        #display_step = FLAGS.display_step
        self.model_dir,self.model_folder,self.checkpoint_folder,self.summary_folder=ModelKeeper.get_folders(scope,make=True)
        #self.model_folder= os.path.join(self.model_dir,scope)
        #self.checkpoint_folder=os.path.join(self.model_folder, 'checkpoints')
        #self.summary_folder=os.path.join(self.model_folder, 'summary')
        #make_folders([ self.model_dir, self.model_folder, self.checkpoint_folder,self.summary_folder] )
        self.cnt_epoch,self.cnt_epoch_add_one= tf_counter('counter_train_epochs')
        self.cnt_step,self.cnt_step_add_one= tf_counter('counter_train_steps')
        #self.train_step=tf.Variable(0, name='train_step', trainable=False)
        #self.inc_step=tf.assign(self.train_step, self.train_step+1 )
        #self.num_threads=FLAGS.num_threads
        #if inputs is not None:
        #    try:
        #            self.train_inputs, self.test_inputs= inputs
        #    except:
        #            self.train_inputs=inputs

        self._define_graphs(**kwargs)

        with tf.control_dependencies( [self.cnt_step_add_one] ):
            self.train_op=tf.group( self.train_graph.train_op )

        self.coord = tf.train.Coordinator()

        #writer
        self.writer = tf.train.SummaryWriter(self.summary_folder ,self.session.graph)# graph_def depricated
        self.merged=tf.merge_all_summaries()

        with tf.device('/cpu:0'):
            #saver has to be constructed after the whole graph is built
            self.save_vars=[v for v in tf.all_variables() if not 'input_producer' in v.name]
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

    def _predict_proba(self,x=None):#must fit in gpu memory at once
        if x is not None:
            fd={self.default_test_graph.x:x}
        else:
            fd=None
        class_prob=self.session.run( self.default_test_graph.class_probs, feed_dict=fd)
        return class_prob[:,1]
    def predict_proba(self,inputs=None):
        P=[]
        for data in chunks(inputs,chunk_size=256):
            class_prob= self._predict_proba(data)
            P.append(class_prob)
        return np.concatenate(P)

    def _predict(self,x=None):#must fit in gpu memory at once
        if x is not None:
            fd={self.default_test_graph.x:x}
        else:
            fd=None
        #y_hat=self.session.run(self.test_graph.y_hat,feed_dict=fd)
        y_hat,class_prob=self.session.run([self.default_test_graph.y_hat,
                                           self.default_test_graph.class_probs],
                                           feed_dict=fd)
        return y_hat ,class_prob[:,1]

    def predict(self,inputs=None):
        Y=[]; P=[]
        i=1
        for data in chunks(inputs,chunk_size=256):
            i+=1
            y_hat,class_prob= self._predict(data)
            Y.append(y_hat)
            P.append(class_prob)
        #print 'test iter',i
        return np.concatenate(Y),np.concatenate(P)

    def evaluate(self,test_inputs=None):
        testX,testY=test_inputs
        y_hat,y_pred=self.predict(testX)
        test_acc=float(np.sum(np.equal(y_hat,testY)))/np.size(testY)
        print 'Test Accuracy: ', test_acc
        false_positive_rate, true_positive_rate, thresholds = roc_curve(testY, y_pred)#actual,predictions
        #false_positive_rate, true_positive_rate, thresholds = roc_curve(testY, y_hat)#actual,predictions
        #print 'False Positive Rate: ',false_positive_rate#just prints matrices
        #print 'True Positive Rate: ',true_positive_rate
        roc_auc = auc(false_positive_rate, true_positive_rate)
        print 'AUC: ', roc_auc

        eval_dict=dict(auc=roc_auc,
            train_pred=np.mean(y_hat),
            tpr=np.median(true_positive_rate),
            fpr=np.median(false_positive_rate),
            acc=test_acc)
        summ = tf.Summary(value=[tf.Summary.Value(tag='test_'+key,simple_value=float(value)) for key,value in eval_dict.iteritems()])
        self.write([summ])
        #eval_summary= tf.Summary(summ)
        return eval_dict

    def fit(self,train_inputs,n_epochs,batch_size):
        g=self.default_train_graph
        trainX,trainY=train_inputs
        shuffle_ind = np.mgrid[0:len(trainX)]
        for e in range(n_epochs):

            t0=time.time()
            #logger=Accumulator( g.log_names )
            logger=Accumulator(['train_acc','train_loss','train_pred'])
            #ave_hat=0
            np.random.shuffle( shuffle_ind )#inplace operation
            trX=trainX[shuffle_ind]; trY=trainY[shuffle_ind]
            for x_batch,y_batch in chunks([trX,trY],chunk_size=256,axis=1):
                x_batch=x_batch.astype('float')
                y_batch=np.transpose( np.vstack([1-y_batch,y_batch] ) )
                feed_dict = {g.x: x_batch, g.y: y_batch}
                #acc, _ = self.session.run([g.accuracy, self.train_op], feed_dict=feed_dict)

                #output=self.session.run([self.train_op]+[self.merged]+g.log_tensors], feed_dict=feed_dict)
                #summ=output[1]
                #logger.log([output[2:])
                ##TODO: self.train_op not used here
                if self.merged is not None:
                    summ, acc,loss,y_hat, _ = self.session.run([self.merged, g.accuracy,g.loss,g.y_hat, g.train_op], feed_dict=feed_dict)
                else:#no summaries created
                    acc,loss,y_hat, _ = self.session.run([ g.accuracy,g.loss,g.y_hat, g.train_op], feed_dict=feed_dict)

                logger.log([acc, loss, np.mean(y_hat)], weight=x_batch.shape[0])#has to be correct order
                #log_pred.append(np.mean(y_hat))#only works for 2 classes
                #log_acc.append(acc); log_loss.append(loss); log_batch_size.append(x_batch.shape[0])

            epoch_time=time.time()-t0
            if epoch_time>1:
                print 'epoch complete'
            ##Summarize and save per epoch
            #batch_sz=np.array(log_batch_size)

            #eval_dict=dict(train_acc=np.average( log_acc, weights=batch_sz ),
            #            train_loss=np.average( log_loss, weights=batch_sz ),
            #            train_classpred=np.average( log_pred,weights=batch_sz))

            eval_dict=logger.reduce_mean()

            train_summ = tf.Summary(value=[tf.Summary.Value(tag=key,simple_value=value) for key,value in eval_dict.iteritems()])

            self.write([train_summ])
            if self.merged is not None:
                self.write([summ])
            #self.save()
            print 'epoch:',self.epoch,' train_acc:%.5f' % eval_dict['train_acc'],'train_loss: ',eval_dict['train_loss']
            #print 'epoch:',self.epoch,' batch_acc:%.5f' % acc,' loss: ',loss
            self.session.run(self.cnt_epoch_add_one)

