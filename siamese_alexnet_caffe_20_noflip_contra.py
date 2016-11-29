from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time
import math

from numpy import *
from utils import *
from PIL import Image
from metric import *
from prepare_data import *
import numpy as np
import random
from six.moves import urllib
from six.moves import xrange    # pylint: disable=redefined-builtin
import tensorflow as tf
import pdb

CAFFE_ALEXNET_PARA = load('bvlc_alexnet.npy').item()

IMAGE_SIZE = 220
NUM_CHANNELS = 3
SEED = 66478    # Set to None for random seed.
NUM_EPOCHS = 100
BATCH_SIZE = 128
EVAL_FREQUENCY = 10    # Number of steps between evaluations.
MODEL_SAVED_FREQUENCY = 200    # Number of steps between saving model.
KERNEL_SAVED_FREQUENCY = 200    # Number of steps between saving kernel map and activaitions.

tf.app.flags.DEFINE_boolean('use_fp16', False, "Use half floats instead of full floats if True.")
tf.app.flags.DEFINE_string('data_dir', './img/query_sku_img_all_cid_filter_dir', 'Directory for storing data')
tf.app.flags.DEFINE_string('debug_dir', './model_alexnet_caffe_20_noflip_contra', 'Debugging directory')
FLAGS = tf.app.flags.FLAGS
if not os.path.exists(FLAGS.debug_dir):
    os.mkdir(FLAGS.debug_dir)
if not os.path.exists(FLAGS.debug_dir + '/logs'):
    os.mkdir(FLAGS.debug_dir + '/logs')
if not os.path.exists(FLAGS.debug_dir + '/pred_case'):
    os.mkdir(FLAGS.debug_dir + '/pred_case')

 
def main(argv=None):    
    
    print('begin')
    #######################################################################
    ## data prepare
    print('begin loading')
    images, labels, img_names = read_img_for_train_multi_thread(FLAGS.data_dir, one_hot=False, multi_channel=True, THREAD_NUM=100, img_num=sys.maxint, flip_img=False)
    #images, labels, img_names = read_img_for_train_multi_thread(FLAGS.data_dir, one_hot=False, multi_channel=True, THREAD_NUM=100, img_num=10000, flip_img=False)
    size = len(images)
    print('loaded ' + str(size) + ' imgs')
    
    train_interval = int(size*0.8)
    val_interval = int(size*0.83)

    train_data = images[:train_interval]
    train_labels = labels[:train_interval]
    train_names = img_names[:train_interval]
    val_data = images[train_interval:val_interval] 
    val_labels = labels[train_interval:val_interval]
    val_names = img_names[train_interval:val_interval]
    test_data = images[val_interval:] 
    test_labels = labels[val_interval:]  
    test_names = img_names[val_interval:]  
        
        
    TRAIN_SIZE = len(train_data)
    IMAGE_SIZE = len(train_data[0]) 
    NUM_CHANNELS = len(train_data[0][0][0])
    
    print('train data shape: ', len(train_data))
    print('validation data shape: ', len(val_data))
    print('test data shape: ', len(test_data))
    print('img size: ', IMAGE_SIZE)
    print('img channel: ', NUM_CHANNELS)

    # create img pairs
    tr_pairs, tr_y, tr_n = create_pairs(train_data,  train_labels, train_names, one_pair_num=1)
    va_pairs, va_y, va_n = create_pairs(val_data, val_labels, val_names, one_pair_num=1)
    te_pairs, te_y, te_n = create_pairs(test_data, test_labels, test_names, one_pair_num=1)
    
    print('train data pair: ', len(tr_pairs))
    print('validation data pair: ', len(va_pairs))
    print('test data pair: ', len(te_pairs))

    # data feed
    train_data_node_L = tf.placeholder(data_type(),
            shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
            name='L')
    train_data_node_R = tf.placeholder(data_type(),
            shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
            name='R')
    train_labels_node = tf.placeholder(data_type(),
            shape=(BATCH_SIZE,1),
            name='gt')
    is_train_node = tf.placeholder(tf.int32, name='train_flag')
    #dropout_node = tf.placeholder(tf.float32)

    ############################################################################
    # model parameter   
    # 96 256 384 256 256
    
    kernel1 = tf.Variable(CAFFE_ALEXNET_PARA["conv1"][0], name='caff_alexnet_kernel1',trainable=False)
    biases1 = tf.Variable(CAFFE_ALEXNET_PARA["conv1"][1], name='caff_alexnet_biases1',trainable=False)
    kernel2 = tf.Variable(CAFFE_ALEXNET_PARA["conv2"][0], name='caff_alexnet_kernel2',trainable=False)
    biases2 = tf.Variable(CAFFE_ALEXNET_PARA["conv2"][1], name='caff_alexnet_biases2',trainable=False)
    kernel3 = tf.Variable(CAFFE_ALEXNET_PARA["conv3"][0], name='caff_alexnet_kernel3',trainable=False)
    biases3 = tf.Variable(CAFFE_ALEXNET_PARA["conv3"][1], name='caff_alexnet_biases3',trainable=False)
    kernel4 = tf.Variable(CAFFE_ALEXNET_PARA["conv4"][0], name='caff_alexnet_kernel4',trainable=False)
    biases4 = tf.Variable(CAFFE_ALEXNET_PARA["conv4"][1], name='caff_alexnet_biases4',trainable=False)
    kernel5 = tf.Variable(CAFFE_ALEXNET_PARA["conv5"][0], name='caff_alexnet_kernel5',trainable=False)
    biases5 = tf.Variable(CAFFE_ALEXNET_PARA["conv5"][1], name='caff_alexnet_biases5',trainable=False)
    fc6_weights = tf.Variable(CAFFE_ALEXNET_PARA["fc6"][0], name='caff_alexnet_fc6_weights',trainable=False)
    fc6_biases = tf.Variable(CAFFE_ALEXNET_PARA["fc6"][1], name='caff_alexnet_fc6_biases',trainable=False)
    fc7_weights = tf.Variable(CAFFE_ALEXNET_PARA["fc7"][0], name='caff_alexnet_fc7_weights')
    fc7_biases = tf.Variable(CAFFE_ALEXNET_PARA["fc7"][1], name='caff_alexnet_fc7_biases')
    #fc8_weights = tf.Variable(CAFFE_ALEXNET_PARA["fc8"][0], name='caff_alexnet_fc8_weights')
    #fc8_biases = tf.Variable(CAFFE_ALEXNET_PARA["fc8"][1], name='caff_alexnet_fc8_biases')
     
    fc8_weights = tf.Variable(    # fully connected, depth 512.
                tf.truncated_normal([4096, 20],
                stddev=0.1,
                seed=SEED,
                dtype=data_type()),name="fc8_weights")
    fc8_biases = tf.Variable(tf.constant(0.1, shape=[20], dtype=data_type()), name="fc8_biases")
    
    parameters = [kernel1, biases1, kernel2, biases2, kernel3, biases3, kernel4, biases4, kernel5, biases5, fc6_weights, fc6_biases, fc7_weights, fc7_biases, fc8_weights, fc8_biases]
    
    for para in parameters:
        print_shape(para)
    
    '''
    variable_summaries(kernel1, 'conv1/weights')
    variable_summaries(kernel2, 'conv2/weights')
    variable_summaries(kernel3, 'conv3/weights')
    variable_summaries(kernel4, 'conv4/weights')
    variable_summaries(kernel5, 'conv5/weights')
    variable_summaries(fc6_weights, 'fc6/weights')
    variable_summaries(fc6_biases, 'fc6/biases')
    '''
    variable_summaries(fc7_weights, 'fc7/weights')
    variable_summaries(fc7_biases, 'fc7/biases')
    variable_summaries(fc8_weights, 'fc8/weights')
    variable_summaries(fc8_biases, 'fc8/biases')
    
    
    ###############################################################################
    # inference functions needed
    def inference_alexnet(images, flag = 'L', _dropout = 0.5):
            
        def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):
            '''
            From https://github.com/ethereon/caffe-tensorflow
            '''
            c_i = input.get_shape()[-1]
            assert c_i%group==0
            assert c_o%group==0
                
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
            if group==1:
                 conv = convolve(input, kernel)
            else:
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                conv = tf.concat(3, output_groups)
                    
            return tf.reshape(tf.nn.bias_add(conv, biases), [-1]+conv.get_shape().as_list()[1:])
                
        
        activations = []
        # conv1
        with tf.variable_scope('conv1') as scope:
            k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
            conv1_in = conv(images, kernel1, biases1, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
            conv1 = tf.nn.relu(conv1_in, name='conv1')
                
            activations += [conv1]
            #tf.histogram_summary('conv1/activations_' + flag, conv1)
    
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1,
                                depth_radius=radius,
                                alpha=alpha,
                                beta=beta,
                                bias=bias)
        # pool1
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        pool1 = tf.nn.max_pool(lrn1,
                                ksize=[1, k_h, k_w, 1],
                                strides=[1, s_h, s_w, 1],
                                padding=padding,
                                name='pool1')
    
        # conv2
        with tf.variable_scope('conv2') as scope:
            k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
            conv2_in = conv(pool1, kernel2, biases2, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv2 = tf.nn.relu(conv2_in, name='conv2')
                
            activations += [conv2]
            #tf.histogram_summary('conv2/activations_' + flag, conv2)
    
        radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2,
                                depth_radius=radius,
                                alpha=alpha,
                                beta=beta,
                                bias=bias)
    
        # pool2
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        pool2 = tf.nn.max_pool(lrn2,
                                ksize=[1, k_h, k_w, 1],
                                strides=[1, s_h, s_w, 1],
                                padding=padding,
                                name='pool2')
    
        # conv3
        with tf.name_scope('conv3') as scope:
            k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
            conv3_in = conv(pool2, kernel3, biases3, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv3 = tf.nn.relu(conv3_in,name='conv3')
                
            activations += [conv3]
            #tf.histogram_summary('conv3/activations_' + flag, conv3)
    
        # conv4
        with tf.name_scope('conv4') as scope:
            k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
            conv4_in = conv(conv3, kernel4, biases4, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv4 = tf.nn.relu(conv4_in, name='conv4')
                
            activations += [conv4]
            #tf.histogram_summary('conv4/activations_' + flag, conv4)
    
        # conv5
        with tf.name_scope('conv5') as scope:
            k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
            conv5_in = conv(conv4, kernel5, biases5, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv5 = tf.nn.relu(conv5_in, name='conv5')
                
            activations += [conv5]
            #tf.histogram_summary('conv5/activations_' + flag, conv5)
        
        # pool5
        k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
        pool5 = tf.nn.max_pool(conv5,
                                ksize=[1, k_h, k_w, 1],
                                strides=[1, s_h, s_w, 1],
                                padding=padding,
                                name='pool5')
    
        pool_shape = pool5.get_shape().as_list()
        pool5 = tf.reshape(pool5,
                                [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
            
        # fc6
        with tf.variable_scope('fc6') as scope:
            hidden6 = tf.nn.relu(tf.matmul(pool5, fc6_weights) + fc6_biases)
            #hidden6 = tf.nn.dropout(hidden6,_dropout)
    
            activations += [hidden6]
            #tf.histogram_summary('fc6/activations_' + flag, hidden6)
                
        # fc7
        with tf.variable_scope('fc7') as scope:
            hidden7 = tf.nn.relu(tf.matmul(hidden6, fc7_weights) + fc7_biases)
            hidden7 = tf.nn.dropout(hidden7,_dropout)
    
            activations += [hidden7]
            #tf.histogram_summary('fc7/activations_' + flag, hidden7)
    
        # fc8  do not have RELU!
        with tf.variable_scope('fc8') as scope:
            hidden8 = tf.matmul(hidden7, fc8_weights) + fc8_biases
    
            #tf.histogram_summary('fc8/activations_' + flag, hidden8)
        
        return hidden8, activations 


    with tf.variable_scope("siamese") as scope: 
        drop_out = 1.0
        if is_train_node == 1:
            drop_out = 0.5
        model1, activations1 = inference_alexnet(train_data_node_L, 'L', drop_out)
        conv1, conv2, conv3, conv4, conv5, hidden6, hidden7 = activations1
        
        scope.reuse_variables()
        model2, activations2 = inference_alexnet(train_data_node_R, 'R', drop_out)
        


    ##############################################################################
    # Small utility function to evaluate a dataset by feeding batches of data to
    # {eval_data} and pulling the results from {eval_predictions}.
    # Saves memory and enables this to run on smaller GPUs.
    def eval_in_batches(tr_pairs, tr_y, sess):
        """Get all predictions for a dataset by running it in small batches."""
        size = len(tr_pairs)
        if size < BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        
        predictions = numpy.ndarray(shape=(size, 1), dtype=numpy.float32)
        auc_list = []
        pair_list = []
        for begin in xrange(0, size, BATCH_SIZE):
            end = begin + BATCH_SIZE
            if end <= size:
                input1,input2,y = next_batch(begin,end,tr_pairs,tr_y)
                feed_dict={train_data_node_L:input1, train_data_node_R:input2, train_labels_node:y, is_train_node:0}
                pred = sess.run(distance, feed_dict=feed_dict)
                predictions[begin:end, :] = pred
                auc_list.append(compute_accuracy(pred, y)) 
                #pair_list.append(compute_accuracy_pair(pred, y))
            else:
                input1,input2,y =next_batch(size-BATCH_SIZE,size,tr_pairs,tr_y)
                feed_dict={train_data_node_L:input1, train_data_node_R:input2, train_labels_node:y, is_train_node:0}
                batch_predictions = sess.run(distance, feed_dict=feed_dict)
                predictions[begin:, :] = batch_predictions[begin - size:, :]
                auc_list.append(compute_accuracy(batch_predictions, y)) 
                #pair_list.append(compute_accuracy_pair(batch_predictions, y))
            
        return predictions, auc_list, pair_list

    ################################################################################
    # loss and optimizaer
    #distance = tf.reduce_sum(model1*model2, 1, keep_dims=True)
    distance = euclid_distance_norm(model1, model2)
    #distance = cosine_distance(model1, model2) 

    tf.histogram_summary('distance' , distance)
    
    #loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(distance, train_labels_node, name='loss'))
    loss = contrastive_loss(train_labels_node, distance) 
    #loss = square_loss(train_labels_node, distance)
    #tf.histogram_summary('loss', loss)
    tf.scalar_summary('loss', loss)

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0, dtype=data_type())
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
            0.01,                                # Base learning rate.
            batch * BATCH_SIZE,    # Current index into the dataset.
            TRAIN_SIZE,                    # Decay step.
            0.95,                                # Decay rate.
            staircase=True)

    opt = tf.train.GradientDescentOptimizer(learning_rate)
    grads = opt.compute_gradients(loss)
    
    grad_list = []
    for grad, var in grads:
        if grad is not None:
            grad_list.append(grad) 
            tf.histogram_summary(var.op.name + '/gradients', grad)
    
    optimizer = opt.apply_gradients(grads, global_step=batch)
    
    

    ##################################################################################
    # begin training
    print('begin training')
    start_time = time.time()
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        
        # Merge all the summaries and write them out to logs
        merged = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(FLAGS.debug_dir + '/logs/train', sess.graph)
        
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()
        print('Initialized!')
        # Loop through training steps.
        for step in xrange(int(NUM_EPOCHS * TRAIN_SIZE) // BATCH_SIZE):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * BATCH_SIZE) % (TRAIN_SIZE - BATCH_SIZE)
            
            input1,input2,y =next_batch(offset,offset + BATCH_SIZE,tr_pairs,tr_y)
            feed_dict={train_data_node_L:input1, train_data_node_R:input2, train_labels_node:y, is_train_node:1}
            
            _,loss_value,lr,predict, summary = sess.run(
                    [optimizer, loss, learning_rate, distance, merged], 
                    feed_dict=feed_dict)
            
             
            if step % EVAL_FREQUENCY == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d (epoch %.2f), %.1f ms' %
                            (step, float(step) * BATCH_SIZE / TRAIN_SIZE,
                             1000 * elapsed_time / EVAL_FREQUENCY))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (loss_value, lr))
                print('Minibatch accuracy, auc: %.3f' %  (compute_accuracy(predict,y)))
                sys.stdout.flush()
                
            if step % (3*EVAL_FREQUENCY) == 0:
                va_p, va_auc, va_pair = eval_in_batches(va_pairs, va_y, sess)
                avg_auc = sum(va_auc)*1.0/len(va_auc)
                #avg_pair = sum(va_pair)*1.0/len(va_pair)
                print('Validation accuracy, auc: %.3f' % (avg_auc))
                
                if step % (10*EVAL_FREQUENCY) == 0:
                    with open(FLAGS.debug_dir + '/pred_case/validation_auc' + str(step), 'w') as file:
                        file.write(' '.join([str(x) for x in va_auc]))
                
                    print_predict_case(step, y, predict, tr_n[offset:(offset + BATCH_SIZE)], va_y, va_p, va_n)
                    train_writer.add_summary(summary, step)
            

            if step % (2*MODEL_SAVED_FREQUENCY) == 0:
                # Finally print the result!
                test_pred, te_auc, te_pair = eval_in_batches(te_pairs, te_y, sess)
                avg_auc = sum(te_auc)*1.0/len(te_auc)
                #avg_pair = sum(te_pair)*1.0/len(te_pair)
                print('Test accuracy, auc: %.3f' % (avg_auc))
            # save model 
            if step % MODEL_SAVED_FREQUENCY == 0:
                save_path = saver.save(sess, FLAGS.debug_dir + "/model_" + str(step) + ".ckpt")
                print("Model saved in file: %s" % save_path)
                
            # output kernel and activations
            if step % KERNEL_SAVED_FREQUENCY == 0:
                pass
            
            '''
                w1 = kernel1.eval()
                w2 = kernel2.eval()
                w3 = kernel3.eval()
                w4 = kernel4.eval()
                w5 = kernel5.eval()
                fc6 = fc6_weights.eval()
                fc7 = fc7_weights.eval()
                fc8 = fc8_weights.eval()

                save_w(w1, 96, 12, 8, FLAGS.debug_dir + '/conv1_'+ str(step) + '.png')
                save_w(w2, 256, 16, 16, FLAGS.debug_dir + '/conv2_'+ str(step) + '.png')
                save_w(w3, 384, 16, 24, FLAGS.debug_dir + '/conv3_'+ str(step) + '.png')
                save_w(w4, 384, 16, 24, FLAGS.debug_dir + '/conv4_'+ str(step) + '.png')
                save_w(w5, 256, 16, 16, FLAGS.debug_dir + '/conv5_'+ str(step) + '.png')
                
                save_fc(fc6, FLAGS.debug_dir + '/fc6_'+ str(step) + '.png')
                save_fc(fc7, FLAGS.debug_dir + '/fc7_'+ str(step) + '.png')
                save_fc(fc8, FLAGS.debug_dir + '/fc8_'+ str(step) + '.png')
                
                feed_dict={train_data_node_L:input1, is_train_node:0}
                cov1, cov2, cov3, cov4, cov5, hi6, hi7, hi8 = sess.run([conv1, conv2, conv3, conv4, conv5, hidden6, hidden7, model1], feed_dict=feed_dict)
                
                print(cov1.shape)
                print(cov2.shape)
                print(cov3.shape)
                print(cov4.shape)
                print(cov5.shape)
                
                #220-->55*55*96-->27*27*256-->13*13*384-->13*13*384-->13*13*384-->6*6*256
                
                save_act(cov1, 64, 16, 16, FLAGS.debug_dir + '/act1_'+ str(step) + '.png') 
                save_act(cov2, 64, 16, 16, FLAGS.debug_dir + '/act2_'+ str(step) + '.png') 
                save_act(cov3, 64, 16, 16, FLAGS.debug_dir + '/act3_'+ str(step) + '.png') 
                save_act(cov4, 64, 16, 16, FLAGS.debug_dir + '/act4_'+ str(step) + '.png') 
                save_act(cov5, 64, 16, 16, FLAGS.debug_dir + '/act5_'+ str(step) + '.png') 
                
                save_hidden(hi6, 64, 8, 8, FLAGS.debug_dir + '/hi6_'+ str(step) + '.png') 
                save_hidden(hi7, 64, 8, 8, FLAGS.debug_dir + '/hi7_'+ str(step) + '.png') 
                save_hidden(hi8, 64, 8, 8, FLAGS.debug_dir + '/hi8_'+ str(step) + '.png') 
                
                
                print_test_img(hi8)            
                '''




    
    
def create_pairs(x, y, name, one_pair_num=1):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    names = []
    class_set = set(y)
    class_num = len(class_set)
    #print(class_set)
    print('different query cnt:', class_num)
    digit_indices = [np.where(np.array(y) == i)[0] for i in class_set]
    for d in range(class_num):
        for i in range(len(digit_indices[d])):
            for idx in range(one_pair_num):
                #pos pair
                if len(digit_indices[d]) ==1:
                    continue
                j = random.randrange(1, len(digit_indices[d]))
                j = (i+j) % len(digit_indices[d])
                z1, z2 = digit_indices[d][i], digit_indices[d][j]
                pairs += [[x[z1], x[z2]]]
                labels += [0]
                names += [[name[z1], name[z2]]]

                #neg pair
                inc = random.randrange(1, class_num)
                dn = (d + inc) % class_num
                j = random.randrange(0, len(digit_indices[dn]))
                z1, z2 = digit_indices[d][i], digit_indices[dn][j]
                pairs += [[x[z1], x[z2]]]
                labels += [1]
                names += [[name[z1], name[z2]]]
    print('pairs cnt: ', len(pairs))
    
    #shuffle data pair
    data_pair_list = []
    for i in range(len(pairs)):
        data_pair_list.append((pairs[i], labels[i], names[i], random.random())) 

    data_pair_list = sorted(data_pair_list, cmp=lambda x,y:cmp(x[3],y[3]))
    
    ret_pairs = []
    ret_labels = []
    ret_names = []
    for i in range(len(data_pair_list)):
        ret_pairs.append(data_pair_list[i][0])  
        ret_labels.append(data_pair_list[i][1])  
        ret_names.append(data_pair_list[i][2])  
    return ret_pairs, ret_labels, ret_names

def contrastive_loss(y,d):
    sim_loss = (1-y) * tf.square(d)
    dissim_loss = y * tf.square(tf.maximum((1 - d),0))
    return tf.reduce_sum(sim_loss + dissim_loss)/2

def square_loss(y,d):
    return tf.reduce_sum(tf.square(y - d)) 


def cosine_distance(y1, y2):
    norm1 = tf.sqrt(tf.reduce_sum(tf.square(y1), 1, keep_dims=True))
    normalized_y1 = y1 / norm1
    
    norm2 = tf.sqrt(tf.reduce_sum(tf.square(y2), 1, keep_dims=True))
    normalized_y2 = y2 / norm2
    
    similarity = tf.reduce_sum(normalized_y1*normalized_y2, 1, keep_dims=True)
    
    return similarity    

def cosine_distance_no_norm(y1, y2):
    
    similarity = tf.reduce_sum(y1*y2, 1, keep_dims=True)

    return similarity    

def euclid_distance_norm(y1, y2):
    norm1 = tf.sqrt(tf.reduce_sum(tf.square(y1), 1, keep_dims=True))
    normalized_y1 = y1 / norm1 + 0.00001
    
    norm2 = tf.sqrt(tf.reduce_sum(tf.square(y2), 1, keep_dims=True))
    normalized_y2 = y2 / norm2
    
    distance = tf.sqrt(tf.reduce_sum(tf.square(normalized_y1-normalized_y2), 1, keep_dims=True))
    
    return distance    


def euclid_distance(y1, y2):
    
    distance = tf.reduce_sum(tf.square(y1-y2), 1, keep_dims=True)
    
    return distance    

def compute_accuracy(prediction,labels):
    
    m = auc_img()
    data = []
    for i in range(len(prediction)):
        data.append((labels[i], prediction[i]))
    return m(data)

def compute_accuracy_pair(prediction,labels):
    
    acc_cnt = 0
    for i in range(0,len(prediction),2):
        if prediction[i]>prediction[i+1] and labels[i]==1 and labels[i+1]==0:
            acc_cnt += 1
        if prediction[i]<prediction[i+1] and labels[i]==0 and labels[i+1]==1:
            acc_cnt += 1
    return acc_cnt*1.0/(len(prediction)/2)
    #return labels[prediction.ravel() < 0.5].mean()
    #return tf.reduce_mean(labels[prediction.ravel() < 0.5])

def next_batch(s,e,inputs,labels):
    inputs_np = numpy.array(inputs[s:e])
    labels_np = numpy.array(labels[s:e])

    input1 = inputs_np[:,0]
    input2 = inputs_np[:,1]
    input1 = input1.astype(numpy.float32)
    input1 = numpy.multiply(input1, 1.0 / 255.0) - 0.5
    input2 = input2.astype(numpy.float32)
    input2 = numpy.multiply(input2, 1.0 / 255.0) - 0.5
    y= np.reshape(labels_np,(len(range(s,e)),1))  
    return input1,input2,y

def data_type():
    """Return the type of the activations, weights, and placeholder variables."""
    if FLAGS.use_fp16:
        return tf.float16
    else:
        return tf.float32

def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

def print_shape(p):
    print(p.name, ' ' , p.get_shape().as_list())

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


def normalize(x):
    x_max = numpy.max(x) 
    x_min = numpy.min(x) 
    x = (x-x_min)*1.0/(x_max-x_min)*255
    return x 

def save_w(w, num, width, height, file_name):
    img = []
    for i in range(num):
        img.append(w[:,:,0:3,i])
    img = numpy.array(img)
    img = normalize(img)
    save_images(img, [width, height], file_name)

def save_fc(w, file_name):
    
    img = numpy.array(w)
    img = normalize(img)
    save_single_images(img, file_name)
    
def save_act(act, num, width, height, file_name):
    img = []
    for i in range(num):
        img.append(normalize(numpy.array(act[i,:,:,0:3])))
        img.append(normalize(numpy.array(act[i,:,:,5:8])))
        img.append(normalize(numpy.array(act[i,:,:,10:13])))
        img.append(normalize(numpy.array(act[i,:,:,15:18])))

    img = numpy.array(img)
    #img = normalize(img)
    save_images(img, [width, height], file_name)

def save_hidden(hidden, num, width, height, file_name):
    img = []
    for i in range(num):
        img.append(normalize(numpy.array(hidden)))

    img = numpy.array(img)
    save_images(img, [width, height], file_name, num_channel=1)


def print_predict_case(step, y, predict, y_na, va_y, va_p, va_na):
    with open(FLAGS.debug_dir + '/pred_case/auc_train_test_score_' + str(step), 'w') as file:
        for i in range(len(y)):
            file.write(str(y[i]) + ' ' + str(predict[i]) + ' ' + '_vs_'.join(y_na[i]) + '\n')
        file.write('\ntest_begin\n')
        for i in range(len(va_y)):
            file.write(str(va_y[i]) + ' ' + str(va_p[i]) + ' ' + '_vs_'.join(va_na[i]) + '\n')

def print_test_img(hidden_pred):
    with open(FLAGS.debug_dir + '/test_img_hidden_file', 'w') as file:
        for i in range(len(hidden_pred)):
            file.write(' '.join([str(x) for x  in hidden_pred[i]]) + '\n')


if __name__ == '__main__':
    tf.app.run()
