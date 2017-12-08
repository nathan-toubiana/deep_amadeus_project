#!/usr/bin/env python
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops



def Model(t_layer_sizes,p_layer_sizes,xs,ys):

    t_input_size = 80

    tf.reset_default_graph()

            #Lstm input data recquires size : batch_size,max_time (spanning back how many time steps), ect..
    
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
    #loss=tf.reduce_mean(loss)
        
        
        
    return outputs_pitch,loss


def cross_entropy(output, input_y):
    with tf.name_scope('cross_entropy'):
        
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=output))

    return ce


def train_step(loss, learning_rate=1e-3):
    with tf.name_scope('train_step'):
        step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    return step


def evaluate(output, input_y):
    with tf.name_scope('evaluate'):
        pred = tf.argmax(output, axis=1)
        error_num = tf.count_nonzero(pred - input_y, name='error_num')
        tf.summary.scalar('LeNet_error_num', error_num)
    return error_num

def training(X_train, y_train,t_layer_sizes,p_layer_sizes,batch_size, pre_trained_model=None):
    
    # define the variables and parameter needed during training
    with tf.name_scope('inputs'):
        xs = tf.placeholder(tf.float32, [None,None, t_input_size])
        ys = tf.placeholder(tf.float32, [None,None, t_input_size])
        
        
    output, loss = Model(t_layer_sizes,p_layer_sizes,xs,ys)

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    step = train_step(loss)
    eve = evaluate(output, ys)

    iter_total = 0
    best_acc = 0
    cur_model_name = 'lenet_{}'.format(int(time.time()))

    with tf.Session() as sess:
        merge = tf.summary.merge_all()

        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # try to restore the pre_trained
        if pre_trained_model is not None:
            try:
                print("Load the model from: {}".format(pre_trained_model))
                saver.restore(sess, 'model/{}'.format(pre_trained_model))
            except Exception:
                print("Load model Failed!")
                pass

        for epc in range(epoch):
            print("epoch {} ".format(epc + 1))

            for itr in range(iters):
                iter_total += 1

                training_batch_x = X_train[itr * batch_size: (1 + itr) * batch_size]
                training_batch_y = y_train[itr * batch_size: (1 + itr) * batch_size]

                _, cur_loss = sess.run([step, loss], feed_dict={xs: training_batch_x, ys: training_batch_y})

                if iter_total % 100 == 0:
                    # do validation
                    valid_eve, merge_result = sess.run([eve, merge], feed_dict={xs: X_val, ys: y_val})
                    valid_acc = 100 - valid_eve * 100 / y_val.shape[0]
                    if verbose:
                        print('{}/{} loss: {} validation accuracy : {}%'.format(
                            batch_size * (itr + 1),
                            X_train.shape[0],
                            cur_loss,
                            valid_acc))

                    # save the merge result summary
                    writer.add_summary(merge_result, iter_total)

                    # when achieve the best validation accuracy, we store the model paramters
                    if valid_acc > best_acc:
                        print('Best validation accuracy! iteration:{} accuracy: {}%'.format(iter_total, valid_acc))
                        best_acc = valid_acc
                        saver.save(sess, 'model/{}'.format(cur_model_name))

    print("Traning ends. The best valid accuracy is {}. Model named {}.".format(best_acc, cur_model_name))

        
        
        
        




