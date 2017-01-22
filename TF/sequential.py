import re
import sys
import csv
import numpy as np
import pandas as pd
import datetime
import os
from dateutil.relativedelta import relativedelta
#from tkrModel import gbm, tree, nnet
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf


from TF.tf_basic import Basic_Graph, Basic_Model

class SeqGraph(Basic_Graph):
    logits_nz=None
    y_nz=None
    def _loss_graph(self):
        #distinguish between:
        #	y:      [batch,n_steps, n_classes]
        #	y_nz:   [nonzero seq elements, n_classes]
        #	y_flat:	[batch*n_steps,n_classes]
        #

        if self.logits_nz is not None:
            logits=self.logits_nz
        else:
            logits=self.logits
        if self.y_nz is not None:
            y=self.y_nz
        else:
            y=self.y


        self.cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits, y, name='cross_entropy') # (n_steps x batch_size,) 
        self.loss=tf.reduce_mean(self.cross_entropy, name='loss')
        ## Optional regularization that will occur if defined self.lambda ##
        if hasattr(self,'TF_lambdaL2'):
            weights= tf.get_collection(tf.GraphKeys.WEIGHTS)
            #l2_reg= tf.reduce_sum(tf.pack([tf.l2_loss(w) for w in weights]))
            l2_reg=tf.add_n([tf.nn.l2_loss(w) for w in weights])
            self.loss+= self.TF_lambdaL2* l2_reg

        #Spelled out version of the below:
        #self.opt= tf.train.AdamOptimizer(self.learning_rate)
        #self.gradients=self.opt.compute_gradients(self.loss)
        #self.train_op=self.opt.apply_gradients(self.gradients)

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def _analysis_graph(self):
        self.y1d=tf.squeeze(tf.split( split_dim=1, num_split=2, value=self.y_nz )[1])
        self.is_correct= tf.equal(  tf.cast(self.y_hat_nz,tf.int64),
                            tf.cast(self.y1d,tf.int64)  )
        self.num_correct= tf.reduce_sum( tf.cast( self.is_correct, tf.float32) )
        self.num_possible=tf.size(self.y1d)
        self.accuracy=tf.div( tf.cast(self.num_correct,tf.float32), tf.cast(self.num_possible,tf.float32) )
        #self.testing_targets=[self.num_possible, self.num_correct, self.y_2d, self.class_probs]

    def _summaries_graph(self):
        #most of this is a copy of tf_basic
        #some notation needs changing for sequential
        #some additional summaries may be added

        '''create summaries for graph'''
        ####If these things are defined somewhere in your graph this could be useful####
        #prefix=''
        col=[self.summaries]
        #    prefix+='train'
        if self.is_training:#NOTE - do we need this part
            if hasattr(self, 'loss'):
                tf.summary.scalar('batch_loss', self.loss,col)
                stream_loss,update_loss= tf.contrib.metrics.streaming_mean(
                    self.loss,
                    metrics_collections=self.metrics,
                    updates_collections=self.update_ops)
                tf.summary.scalar(self.prefix+'_'+'loss',stream_loss,col)
            #Accuracy
            if hasattr(self, 'accuracy'):
                tf.summary.scalar('batch_accuracy', self.accuracy,col)
                stream_acc,update_acc= tf.contrib.metrics.streaming_accuracy(
                    predictions=self.y_hat_nz,labels=self.y1d)
                tf.add_to_collection(self.update_ops,update_acc)
                tf.add_to_collection(self.metrics,stream_acc)
                tf.summary.scalar(self.prefix+'_'+'acc',stream_acc,col)

            if hasattr(self, 'logits'):
                tf.summary.histogram('batch_logits', self.logits_nz,col)

            if hasattr(self,'class_probs'):
                tf.summary.histogram('class_probs', self.class_probs_nz,col)
            #AUC
            if hasattr(self,'y_hat') and self.n_classes == 2:
                #doc says labels have to be bool. check results
                self.bool_y=tf.cast(self.y1d, tf.float32)
                self.bool_y_hat=tf.cast(self.y_hat_nz, tf.float32)
                stream_auc,update_auc= tf.contrib.metrics.streaming_auc(
                    predictions=self.bool_y_hat,labels=self.bool_y,
                    metrics_collections=self.metrics,
                    updates_collections=self.update_ops)
                tf.summary.scalar(self.prefix+'_'+'auc',stream_auc,col)


