import os
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('num_threads',4,'Number of cpu threads to use on batch queue')
flags.DEFINE_integer('num_epochs', 1, 'Number of epochs to run trainer.')
#flags.DEFINE_integer('num_iter', None, 
flags.DEFINE_string('train_dir', '/home/chris/models/train_data', 'Directory with the training data.')
flags.DEFINE_integer('n_hidden',64,'state/output size of rnn')
flags.DEFINE_integer('batch_size',128,'Batch Size.')
flags.DEFINE_integer('n_input',2308,'dim of time dep feature')
flags.DEFINE_integer('n_context',15,'dim of time ind feature')
flags.DEFINE_integer('n_steps',10,'max number of steps in data')
flags.DEFINE_integer('capacity',100,'number of samples to store in batch queue')
flags.DEFINE_integer('display_step',100,'how often to save and report')
flags.DEFINE_string('model_dir','/home/chris/models','where to save model')
flags.DEFINE_integer('training_iters',-1,'-1 means not specified (rely on num_epoch)')

flags.DEFINE_string('mode','train','possible: train,test, eval')


# Initializing the variables
batch_size = FLAGS.batch_size #256 #64 #
num_threads= FLAGS.num_threads
if FLAGS.training_iters > 0:
	training_iters=FLAGS.training_iters
else:
	training_iters=np.inf

display_step = FLAGS.display_step
n_hidden= FLAGS.n_hidden
n_context=FLAGS.n_context
#n_persistent=len(train.ploc)
n_input = FLAGS.n_input
n_steps = FLAGS.n_steps# timesteps [some sequences truncated]
num_epochs= FLAGS.num_epochs
capacity=FLAGS.capacity


