# -*- coding: utf-8 -*-
'''
Code for ground truth evaluation of sybil detection with photos with AlexNet model

The implementation of finetuning AlexNet model is based on
https://github.com/kratzert/finetune_alexnet_with_tensorflow

* For ground truth evaluation, large vendors will be split into two pesudo vendors, which will be
added into training and testing set respectively
* Network model was pre-trained on ImageNet, and training data are used to finetune the network weights
* Prediction is made on each image in testing set and results are averaged to obtain the
vendor similarities

To use the code, the data folder need to be changed accordingly
'''


import os
import numpy as np
import tensorflow as tf
from misc_files.alexnet import AlexNet
from misc_files.datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator
import sys
from json import load as loadjson
import pickle

"""
Configuration Part.
"""


def alex_tr_pred(setname='Agora', threshold='10', step_='pseudo_pairing',
                 root_dir='/media/intel/m2/train_test_data/',
                 tmp_file='pretrained_models/Alexnet_tmpfile'):

    # Loading class names
    tr_test_path = os.path.join(root_dir, step_, setname, str(threshold), 'labels')
    with open(os.path.join(tr_test_path, 'class_name.json')) as fp:
        class_name = loadjson(fp)

    # Path to the textfiles for the training and testing set
    train_file = os.path.join(tr_test_path, 'train.txt')
    test_file = os.path.join(tr_test_path, 'test.txt')

    # Learning params
    learning_rate = 0.01
    num_epochs = 30
    batch_size = 128

    # Network params
    dropout_rate = 0.5
    num_classes = len(class_name)
    train_layers = ['fc8', 'fc7', 'fc6']

    # How often we want to write the tf.summary data to disk
    display_step = 20

    # Path for tf.summary.FileWriter and to store model checkpoints
    filewriter_path = os.path.join(tmp_file, "tensorboard")
    checkpoint_path = os.path.join(tmp_file, "checkpoints")

    try:
        os.makedirs(filewriter_path)
    except:
        pass
    try:
        os.makedirs(checkpoint_path)
    except:
        pass

    """
    Main Part of the finetuning Script.
    """

    # Create parent path if it doesn't exist
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)

    # Place data loading and preprocessing on the cpu
    with tf.device('/cpu:0'):
        tr_data = ImageDataGenerator(train_file,
                                     mode='training',
                                     batch_size=batch_size,
                                     num_classes=num_classes,
                                     shuffle=True)
        test_data = ImageDataGenerator(test_file,
                                       mode='inference',
                                       batch_size=batch_size,
                                       num_classes=num_classes,
                                       shuffle=True)

        # create an reinitializable iterator given the dataset structure
        iterator = Iterator.from_structure(tr_data.data.output_types,
                                           tr_data.data.output_shapes)
        next_batch = iterator.get_next()

    # Ops for initializing the two different iterators
    training_init_op = iterator.make_initializer(tr_data.data)
    test_init_op = iterator.make_initializer(test_data.data)

    # TF placeholder for graph input and output
    x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
    y = tf.placeholder(tf.float32, [batch_size, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    # Initialize model
    model = AlexNet(x, keep_prob, num_classes, train_layers)

    # Link variable to model output
    score = model.fc8
    softmax = tf.nn.softmax(score)
    # List of trainable variables of the layers we want to train
    var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

    # Op for calculating the loss
    with tf.name_scope("cross_ent"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                      labels=y))

    # Train op
    with tf.name_scope("train"):
        # Get gradients of all trainable variables
        gradients = tf.gradients(loss, var_list)
        gradients = list(zip(gradients, var_list))

        # Create optimizer and apply gradient descent to the trainable variables
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients)

    # Add gradients to summary
    for gradient, var in gradients:
        tf.summary.histogram(var.name + '/gradient', gradient)

    # Add the variables we train to the summary
    for var in var_list:
        tf.summary.histogram(var.name, var)

    # Add the loss to summary
    tf.summary.scalar('cross_entropy', loss)

    # Merge all summaries together
    merged_summary = tf.summary.merge_all()

    # Initialize the FileWriter
    writer = tf.summary.FileWriter(filewriter_path)

    # Initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # Get the number of training/validation steps per epoch
    train_batches_per_epoch = int(np.floor(tr_data.data_size / batch_size))
    test_batches_per_epoch = int(np.ceil(test_data.data_size / batch_size))

    # Start Session
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())

    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)

    # Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    # Loop over number of epochs
    for epoch in range(num_epochs):

        print("Epoch number: %d" %(epoch + 1))

        # Initialize iterator with the training dataset
        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            # get next batch of data
            img_batch, label_batch = sess.run(next_batch)

            # And run the training op
            sess.run(train_op, feed_dict={x: img_batch,
                                          y: label_batch,
                                          keep_prob: dropout_rate})

            # Generate summary with the current batch of data and write to file
            if step % display_step == 0:
                s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                        y: label_batch,
                                                        keep_prob: 1.})

                writer.add_summary(s, epoch * train_batches_per_epoch + step)

        print("Saving checkpoint of model")

        # save checkpoint of the model
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name)

        print("Checkpoint saved at {}".format(datetime.now(), checkpoint_name))
    #
    #
    # Prediction

    final_pred = {}
    sess.run(test_init_op)
    for j in range(test_batches_per_epoch):
        img_batch, label_batch = sess.run(next_batch)
        if j == 0:
            _temp_img_batch, _temp_label_batch = img_batch, label_batch
        elif j == test_batches_per_epoch - 1:
            img_batch = np.concatenate((img_batch, _temp_img_batch), axis=0)[:batch_size]
        probs = sess.run(softmax, feed_dict={x: img_batch, keep_prob: 1})[:len(label_batch)]
        for i in range(len(label_batch)):
            img_class = list(label_batch[i]).index(1.)
            if img_class not in final_pred:
                final_pred[img_class] = []
            final_pred[img_class].append(list(probs[i]))
    
    final_pred_path = os.path.join(tr_test_path, '../final_pred', 'AlexNet')
    try:
        os.makedirs(final_pred_path)
    except:
        pass
    with open(os.path.join(final_pred_path, 'prob.plk'), 'wb') as fp:
        pickle.dump(final_pred, fp)
        
    test_ct = len(final_pred.keys())
    corr_ct = 0.
    for k in final_pred.keys():
        pred_class = np.argmax(np.array(final_pred[k]).mean(axis=0))
        if k == pred_class:
            corr_ct += 1
        else:
            print("%s <xxx> %s" % (class_name[str(k)], class_name[str(pred_class)]))
    print(test_ct, int(corr_ct), corr_ct * 100. / test_ct)
    
    sess.close()

if __name__ == '__main__': 
    alex_tr_pred(setname=sys.argv[1], threshold=sys.argv[2], step_=sys.argv[3])
