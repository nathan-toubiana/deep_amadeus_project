#!/usr/bin/env python
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def Model(t_layer_sizes,p_layer_sizes):

    t_input_size = 80

    tf.reset_default_graph()

            #Lstm input data recquires size : batch_size,max_time (spanning back how many time steps), ect..
    xs = tf.placeholder(tf.float32, [None,None, t_input_size])
    ys = tf.placeholder(tf.float32, [None,None, t_input_size])
            #xs = tf.one_hot(xss, depth=1000, axis=-1)
            #xs_onehot = tf.one_hot(xs, depth=1000, axis=-1)

            # From our architecture definition, size of the notewise input

            # time network maps from notewise input size to various hidden sizes
    lstm_time=[]

    for i in t_layer_sizes:
        lstm_time.append(tf.contrib.rnn.LSTMCell(i))

    time_model=tf.contrib.rnn.MultiRNNCell(lstm_time)        
    init_state_time=time_model.zero_state(tf.shape(ys)[0],tf.float32)
    with tf.variable_scope('lstm1'):
        outputs_time,final_state_time=tf.nn.dynamic_rnn(time_model, xs, initial_state = init_state_time, dtype = tf.float32)
            #self.time_model = StackedCells( self.t_input_size, celltype=LSTM, layers = t_layer_sizes)
            #self.time_model.layers.append(PassthroughLayer())

            # pitch network takes last layer of time model and state of last note, moving upward
            # and eventually ends with a two-element sigmoid layer

    p_input_size = t_layer_sizes[-1] + 2


    lstm_pitch=[]

    for i in p_layer_sizes:
        lstm_pitch.append(tf.contrib.rnn.LSTMCell(i))
    lstm_pitch.append(tf.contrib.rnn.LSTMCell(80))


    pitch_model=tf.contrib.rnn.MultiRNNCell(lstm_pitch)

    init_state_pitch=pitch_model.zero_state(tf.shape(ys)[0],tf.float32)
    with tf.variable_scope('lstm2'):
        outputs_pitch,final_state_pitch=tf.nn.dynamic_rnn(pitch_model,outputs_time,initial_state = init_state_pitch,dtype = tf.float32)

    loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=ys,logits=outputs_pitch)
    loss=tf.reduce_mean(loss)
        
        
        
    return outputs_pitch

        
        
        
        
"""    
    
def setup_train(self):
        # dimensions: (batch, time, notes, input_data) with input_data as in architecture
        
        #self.input_mat = tf.placeholder(???)
        
        self.input_mat = T.btensor4()
        # dimensions: (batch, time, notes, onOrArtic) with 0:on, 1:artic
        self.output_mat = T.btensor4()
        #self.output_mat = tf.placeholder(???)
        
        self.epsilon = np.spacing(np.float32(1.0))

        def step_time(in_data, *other):
            other = list(other)
            split = -len(self.t_layer_sizes) if self.dropout else len(other)
            hiddens = other[:split]
            masks = [None] + other[split:] if self.dropout else []
            new_states = self.time_model.forward(in_data, prev_hiddens=hiddens, dropout=masks)
            return new_states
        
        def step_note(in_data, *other):
            other = list(other)
            split = -len(self.p_layer_sizes) if self.dropout else len(other)
            hiddens = other[:split]
            masks = [None] + other[split:] if self.dropout else []
            new_states = self.pitch_model.forward(in_data, prev_hiddens=hiddens, dropout=masks)
            return new_states
        
        # We generate an output for each input, so it doesn't make sense to use the last output as an input.
        # Note that we assume the sentinel start value is already present
        # TEMP CHANGE: NO SENTINEL
        input_slice = self.input_mat[:,0:-1]
        n_batch, n_time, n_note, n_ipn = input_slice.shape
        
        # time_inputs is a matrix (time, batch/note, input_per_note)
        time_inputs = input_slice.transpose((1,0,2,3)).reshape((n_time,n_batch*n_note,n_ipn))
        num_time_parallel = time_inputs.shape[1]
        
        # apply dropout
        if self.dropout > 0:
            time_masks = theano_lstm.MultiDropout( [(num_time_parallel, shape) for shape in self.t_layer_sizes], self.dropout)
        else:
            time_masks = []

        time_outputs_info = [initial_state_with_taps(layer, num_time_parallel) for layer in self.time_model.layers]
        #???
        time_result, _ = theano.scan(fn=step_time, sequences=[time_inputs], non_sequences=time_masks, outputs_info=time_outputs_info)
        
        self.time_thoughts = time_result
        
        # Now time_result is a list of matrix [layer](time, batch/note, hidden_states) for each layer but we only care about 
        # the hidden state of the last layer.
        # Transpose to be (note, batch/time, hidden_states)
        last_layer = get_last_layer(time_result)
        n_hidden = last_layer.shape[2]
        time_final = get_last_layer(time_result).reshape((n_time,n_batch,n_note,n_hidden)).transpose((2,1,0,3)).reshape((n_note,n_batch*n_time,n_hidden))
        
        # note_choices_inputs represents the last chosen note. Starts with [0,0], doesn't include last note.
        # In (note, batch/time, 2) format
        # Shape of start is thus (1, N, 2), concatenated with all but last element of output_mat transformed to (x, N, 2)
        start_note_values = T.alloc(np.array(0,dtype=np.int8), 1, time_final.shape[1], 2 )
        correct_choices = self.output_mat[:,1:,0:-1,:].transpose((2,0,1,3)).reshape((n_note-1,n_batch*n_time,2))
        note_choices_inputs = T.concatenate([start_note_values, correct_choices], axis=0)
        
        # Together, this and the output from the last LSTM goes to the new LSTM, but rotated, so that the batches in
        # one direction are the steps in the other, and vice versa.
        note_inputs = T.concatenate( [time_final, note_choices_inputs], axis=2 )
        num_timebatch = note_inputs.shape[1]
        
        # apply dropout
        if self.dropout > 0:
            pitch_masks = theano_lstm.MultiDropout( [(num_timebatch, shape) for shape in self.p_layer_sizes], self.dropout)
        else:
            pitch_masks = []

        note_outputs_info = [initial_state_with_taps(layer, num_timebatch) for layer in self.pitch_model.layers]
        #???
        note_result, _ = theano.scan(fn=step_note, sequences=[note_inputs], non_sequences=pitch_masks, outputs_info=note_outputs_info)
        
        self.note_thoughts = note_result
        
        # Now note_result is a list of matrix [layer](note, batch/time, onOrArticProb) for each layer but we only care about 
        # the hidden state of the last layer.
        # Transpose to be (batch, time, note, onOrArticProb)
        note_final = get_last_layer(note_result).reshape((n_note,n_batch,n_time,2)).transpose(1,2,0,3)
        
        # The cost of the entire procedure is the negative log likelihood of the events all happening.
        # For the purposes of training, if the ouputted probability is P, then the likelihood of seeing a 1 is P, and
        # the likelihood of seeing 0 is (1-P). So the likelihood is (1-P)(1-x) + Px = 2Px - P - x + 1
        # Since they are all binary decisions, and are all probabilities given all previous decisions, we can just
        # multiply the likelihoods, or, since we are logging them, add the logs.
        
        # Note that we mask out the articulations for those notes that aren't played, because it doesn't matter
        # whether or not those are articulated.
        # The padright is there because self.output_mat[:,:,:,0] -> 3D matrix with (b,x,y), but we need 3d tensor with 
        # (b,x,y,1) instead
        active_notes = T.shape_padright(self.output_mat[:,1:,:,0])
        mask = T.concatenate([T.ones_like(active_notes),active_notes], axis=3)
        
        loglikelihoods = mask * T.log( 2*note_final*self.output_mat[:,1:] - note_final - self.output_mat[:,1:] + 1 + self.epsilon )
        self.cost = T.neg(T.sum(loglikelihoods))
        
        updates, _, _, _, _ = create_optimization_updates(self.cost, self.params, method="adadelta")
        self.update_fun = theano.function(
            inputs=[self.input_mat, self.output_mat],
            outputs=self.cost,
            updates=updates,
            allow_input_downcast=True)

        self.update_thought_fun = theano.function(
            inputs=[self.input_mat, self.output_mat],
            outputs= ensure_list(self.time_thoughts) + ensure_list(self.note_thoughts) + [self.cost],
            allow_input_downcast=True)
    
    
    
    
def training(X_train,Y_train):
   




