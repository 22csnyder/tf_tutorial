"""
Date: June 20, 2016
Author: Chris Snyder

This is a script I wrote that shows you how to use the
classes and methods in the TF folder.
After messing around with Tensorflow for a couple months,
this is the model paradigm that seemed to work in most cases,
and I use it to make writing my own models easier, but it
may not be to everyone's taste.

To run this demo, type 'python nn_model.py' in the command line.
This should create all the necessary folders to save.
Running the script again will pickup training from where the model left off
for the number of epochs specified.
"""



import numpy as np
import time,os
import tensorflow as tf
from tensorflow.python.ops import math_ops,init_ops
from tensorflow.python.ops import variable_scope as vs
from sklearn.cross_validation import train_test_split
from tensorflow.python.ops import math_ops,init_ops

#so it can be run iteratively from ipython
tf.reset_default_graph()
print 'Resetting TFGraph!'

#Some custom 1-line layers I wrote
from TF.utils import _simple_tanh_layer, _simple_sigm_layer, _linear

#Generally you will need to inherit Basic_Graph to write your own graph class
#Basic_Model you can usually get away with using as is
#That is what I do below..
from TF.tf_basic import Basic_Graph, Basic_Model

class NN_Graph(Basic_Graph):
    def __init__(self, n_features,learning_rate=0.01, n_hidden=5, is_training=False):
        self.is_training=is_training
        self.learning_rate=learning_rate
        self.n_features=n_features
        self.n_hidden=n_hidden

        self.x =tf.placeholder("float", [None, self.n_features], name="x")
        self.y =tf.placeholder(tf.int64,  name="y")

        #Define graph
        self._inference_graph()
        self._analysis_graph()
        if self.is_training:
            self._loss_graph()
        self._summaries_graph()


    def _inference_graph(self):
        x_ =_simple_tanh_layer( self.x , output_size=self.n_hidden , scope='embedding')
        x_ =_simple_tanh_layer( x_ ,scope='layer1', summary=self.is_training)
        x_ =_simple_tanh_layer( x_ ,scope='layer2', summary=self.is_training)
        self.logits = _linear( x_ ,output_size=self.n_classes, scope='projection')
        self.class_probs=tf.nn.softmax( self.logits )
        self.y_hat= tf.argmax( self.class_probs, 1)


    #These next two methods are defined in Basic_Graph but written here for understanding
    ##This method is generic so is defined in Basic_Graph###
    #def _analysis_graph(self):
    #    y1d=tf.squeeze(tf.split( split_dim=1, num_split=2, value=self.y )[1])
    #    self.is_correct= tf.equal( self.y_hat, y1d )
    #    self.y1d=y1d
    #    self.num_correct= tf.reduce_sum( tf.cast( self.is_correct, tf.float32) )
    #    self.num_possible=tf.size(y1d)
    #    self.accuracy=tf.div( tf.cast(self.num_correct,tf.float32), tf.cast(self.num_possible,tf.float32) )

	###This method is inherited from Basic_Graph###
    #def _loss_graph(self):
    #    self.cross_entropy=tf.nn.softmax_cross_entropy_with_logits(self.logits, self.y, name='cross_entropy') # (n_steps x batch_size,) 
    #    self.loss=tf.reduce_mean(self.cross_entropy, name='loss')
    #    self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


if __name__=='__main__':
    ##Define some Test data:
    batch_size=512
    learning_rate= 0.001

    np.random.seed(22)
    dataX=np.random.rand(3000,2)
    score=np.sum(dataX**2,axis=1)

    dataY= score < 0.25
    dataY=dataY.astype('int')#must by int64
    train_x, test_x, train_y, test_y = train_test_split(dataX,dataY, test_size=0.1, random_state=22)
    n_features=train_x.shape[1]




    #These are the arguments passed to the inference graph, not the model 
    graph_kwargs=dict(
        learning_rate=learning_rate,
        n_hidden=5,
        n_features=n_features,
    )

    sess=tf.Session()
    #This is the contruction of the model
    #The model handles saving and loading and..
    #..makes good guesses at summaries etc
    model=Basic_Model(graph=NN_Graph,
        graph_kwargs=graph_kwargs,
        session=sess,
        scope='demo_NN_Model')

    model.initialize()#instead of sess.run(tf.init..)

    #This will check for model, and load if /models/scope exists
    model.load_model()

    train_inputs=dict(x=train_x,y=train_y)
    test_inputs=dict(x=test_x,y=test_y)
    train_epochs=5

    train_kwargs={'inputs':train_inputs,
                  'n_epochs':train_epochs,
                  'epoch_per_summary':1,
                  'epoch_per_checkpoint':2,
                  'batch_size':batch_size
                 }

    for i in range(20):
        model.fit(**train_kwargs)
        model.evaluate(test_inputs)












    ##I find the following helpful for debugging
    _x= test_x[:10]
    _y= test_y[:10]
    def check(attr,x=_x,y=_y):
        try:
            return model.session.run( getattr(model.test_graph, attr), feed_dict={model.test_graph.x: x, model.test_graph.y: y })
        except:
            return model.session.run( getattr(model.train_graph, attr), feed_dict={model.train_graph.x: x, model.train_graph.y: y })


