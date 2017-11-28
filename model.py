#!/usr/bin/env python
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


class Model(object):
    """
    Class to define the architecture of the model
    """
    
    def __init__(self, x_input,y_input,t_layer_sizes, p_layer_sizes, dropout=0):
        
        self.t_layer_sizes = t_layer_sizes
        self.p_layer_sizes = p_layer_sizes

        # From our architecture definition, size of the notewise input
        self.t_input_size = 80
        
        # time network maps from notewise input size to various hidden sizes
        lstm_time=[]
        
        for i in t_layer_sizes:
            lstm_time.append(tensorflow.contrib.rnn.LSTMcell(i))
            
        lstm_time=tensorflow.contrib.rnn.MultiRNNCell([lstm])
        self.time_model=lstm_time
        
        init_state_time=lstm_time.zero_state(t_input_size,tf.int32)
        outputs_time,final_state_time=tf.nn.dynamic_rnn(lstm_time,x_input,init_state_time,tf.int32)

        #self.time_model = StackedCells( self.t_input_size, celltype=LSTM, layers = t_layer_sizes)
        #self.time_model.layers.append(PassthroughLayer())

        # pitch network takes last layer of time model and state of last note, moving upward
        # and eventually ends with a two-element sigmoid layer
        
        p_input_size = t_layer_sizes[-1] + 2
        
        
        lstm_pitch=[]
        
        for i in p_layer_sizes:
            lstm_pitch.append(tensorflow.contrib.rnn.LSTMcell(i))
            
        
        lstm_pitch=tensorflow.contrib.rnn.MultiRNNCell([lstm])
        self.pitch_model=lstm_pitch
        
        init_state_pitch=lstm_pitch.zero_state(p_input_size,tf.int32)
        outputs_pitch,final_state_pitch=tf.nn.dynamic_rnn(lstm_pitch,outputs_time,init_state_pitch,tf.int32)

        loss=tf.nn.sigmoid_cross_entropy_with_logits(outputs_pitch,y_input)
        loss=tf.reduce_mean(loss)
        #self.pitch_model = StackedCells( p_input_size, celltype=LSTM, layers = p_layer_sizes)
        #self.pitch_model.layers.append(Layer(p_layer_sizes[-1], 2, activation = T.nnet.sigmoid))
        
        
        
        #self.dropout = dropout

        #self.conservativity = T.fscalar()
        #self.srng = T.shared_randomstreams.RandomStreams(np.random.randint(0, 1024))

        self.setup_train()
        self.setup_predict()
        self.setup_slow_walk()
    
    

    
    
    
    
    
def training(X_train,Y_train):
    """
    training function
    """




