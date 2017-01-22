import re
import time
import sys
import csv
import numpy as np
import pandas as pd
import datetime
import os
from dateutil.relativedelta import relativedelta
#from tkrModel import gbm, tree, nnet
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import roc_curve, auc

import tensorflow as tf
from tf_basic import Basic_Model, Basic_Graph

class Connected_Model(Basic_Model):
    #def fit(self,train_inputs,n_epochs,batch_size):
    def fit(self):
        self.threads=tf.train.start_queue_runners(coord=self.coord,sess=self.session)
        #trainX,trainY=self.train_inputs
        i=0
        log_acc=[]; log_loss=[]; log_batch_size=[];log_pred=[]#Assumes loss is per sample
        self.coord.clear_stop()
        try:
            t1=time.time()
            print 'fitting data!'
            while not self.coord.should_stop():
                #if self.step>=self.training_iters:
                #    print 'training_iters exceeded'
                #    self.coord.request_stop()
                #summ, acc,loss,y_hat, _ = self.session.run([self.merged, self.train_graph.accuracy,self.train_graph.loss,self.train_graph.y_hat, self.train_op])
                bs,acc,loss,y_hat, _ = self.session.run([self.train_graph.batch_size, self.train_graph.accuracy,self.train_graph.loss,self.train_graph.y_hat, self.train_op])

                log_pred.append(np.mean(y_hat))#only works for 2 classes
                log_acc.append(acc); log_loss.append(loss); log_batch_size.append(bs)
                #self.step(inputs)
                i+=1
        except Exception, e:
            t2=time.time()
            print 'num_threads',self.train_stream.num_threads
            print 'batch_size',self.train_stream.batch_size
            print 'n iter=',i
            self.coord.request_stop( e )
        finally:
            if i>0:
                print 'Time per Step:',(t2-t1)/i
                print 'Total Time:',t2-t1#,'\n\t',t1-t0
                #print 'epoch:',e,' batch_acc:%.5f' % acc,' loss: ',loss
                self.coord.request_stop()
                batch_sz=np.array(log_batch_size)
                self.coord.join(self.threads, stop_grace_period_secs=5)
                self.session.run(self.cnt_epoch_add_one)
                eval_dict=dict(train_acc=np.average( log_acc, weights=batch_sz ),
                            train_loss=np.average( log_loss, weights=batch_sz ),
                            train_classpred=np.average( log_pred,weights=batch_sz))
                print 'max batch:',np.max(log_batch_size)
                print 'min batch:',np.min(log_batch_size)
                print 'last batch:',log_batch_size[-1]
                train_summ = tf.Summary(value=[tf.Summary.Value(tag=key,simple_value=value) for key,value in eval_dict.iteritems()])
                #self.write([train_summ])
                #self.write([summ])
                self.save()
            else:
                print 'oops, training never occured. i=',i
    def evaluate(self):
        self.test_threads=tf.train.start_queue_runners(coord=self.coord,sess=self.session)
        i=0
        total_count=0
        total_correct=0
        P=[];A=[]

        self.coord.clear_stop()
        print '#####   Starting testing at epoch ',self.epoch,'  ######'
        try:#with coord.stop_on_exception():#should join at end
            t1=time.time()
            while not self.coord.should_stop():
                n,n_c,np_y_,np_y_hat_=self.session.run( self.test_graph.testing_targets)
                total_count+=n
                total_correct+=n_c
                P.append(np_y_hat_[:,0])
                A.append(np_y_[:,0])
                i+=1
        except Exception, e:
            #e=_filter_exception(e)#in master but not 0.8#not in 0.9
            t2=time.time()
            print 'Finished Testing: n iter=',i
            self.coord.request_stop( e )
            print 'Total Time:',t2-t1
        finally:
            self.coord.request_stop()
            self.coord.join(self.test_threads, stop_grace_period_secs=100)

            if i>0:#loop actually happened
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
                print 'epoch:',self.epoch,' AUC: ', roc_auc
                #return actual, predictions, total_count, total_correct
            else:
                print 'oops, testing never occured. i=',i

    def step(self,inputs=None):
        if inputs is None:
                fd=None
        else:
                raise NotImplementedError
                fd=some_stuff
        self.session.run(self.training_graph.train_op, feed_dict=fd)
        self.global_step +=1
        #self.training_graph.write(fd =fd)

        if self.global_step % self.display_step == 0:
                acc,loss=self.training_graph.write(fd)
                print "Step " + str(self.global_step) + ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                          ", Training Accuracy= " + "{:.5f}".format(acc)
