#Start here
import tensorflow as tf
#from tensorflow.models.rnn import rnn
#from tensorflow.models.rnn.rnn_cell import GRUCell, BasicLSTMCell

#if tf.__version__=='0.9.0rc0':
try:#version 0.9
    GRUCell=tf.nn.rnn_cell.GRUCell
    BasicLSTMCell=tf.nn.rnn_cell.BasicLSTMCell
except:#0.8 and below
    from tensorflow.models.rnn.rnn_cell import GRUCell, BasicLSTMCell

from sklearn.metrics import auc,roc_curve
from sequential_model import RNNModel
from sequential_graph import TrajectoryWrapper
import time,os
from RecordReader import DataStream
import numpy as np

from model_flags import *  #Default behavior and available params
TRAIN_FILE = 'train.tf.records'
TEST_FILE = 'test.tf.records'
#TEST_FILE= 'train.tf.records'#debug

#VALIDATION_FILE = 'validation.tf.records'
train_filename = os.path.join(FLAGS.train_dir,TRAIN_FILE)
test_filename = os.path.join( FLAGS.train_dir,TEST_FILE )
train_data= DataStream( train_filename )
test_data = DataStream( test_filename )

#dropout=0.65

inputs=train_data.batch
t_inputs=test_data.batch

# dgcell= rnn_cell.DropoutWrapper(  rnn_cell.GRUCell( 2*n_hidden ),  input_keep_prob=dropout )
# dlcell=rnn_cell.DropoutWrapper(  rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0), input_keep_prob=dropout )
# tgcell=TrajectoryWrapper(  GRUCell( 2*n_hidden) )
#tlcell=TrajectoryWrapper(  BasicLSTMCell(n_hidden, forget_bias=1.0) )
# lcell=BasicLSTMCell(n_hidden, forget_bias=1.0)

def main(_):
    #run_training()
    session=tf.Session()
    #model_name='lstm_64h'
    #lstm_model=RNNModel(session,'BasicLSTM', uses_context=False,train_inputs=inputs, test_inputs=t_inputs, scope=model_name)
    #model=lstm_model

    #some model??_model=RNNModel(n_input,n_steps,n_hidden,model_name,'BasicLSTM', None)
    time_lstm_model=RNNModel(session, 'BasicLSTM', uses_context=False,train_inputs= inputs,test_inputs=t_inputs, n_input=n_input+1, scope='lstm_64h_wtime')
    model=time_lstm_model

    #model_name='lstm_64h_traj1'
    #with tf.variable_scope(model_name):
    #    traj_lstm_model=RNNModel(n_input,n_steps,n_hidden,model_name,tlcell,n_persistent)

    model.init_variables()
    model.load_model()

    if FLAGS.mode=='train':
        model.fit()
    elif FLAGS.mode=='test':
        model.test_eval()

    session.close()
    print 'Model Setup Complete!'

if __name__=='__main__':
	tf.app.run()
