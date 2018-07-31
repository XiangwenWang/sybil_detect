# -*- coding: utf-8 -*-
'''
Code for ground truth evaluation of sybil detection with photos with Inception model

The implementation of the Inception-V4 model is based on
https://github.com/flyyufelix/cnn_finetune/blob/master/inception_v4.py

* For ground truth evaluation, large vendors will be split into two pesudo vendors, which will be
added into training and testing set respectively
* Network model was pre-trained on ImageNet, and training data are used to finetune the network weights
* Prediction is made on each image in testing set and results are averaged to obtain the
vendor similarities

To use the code, the data folder need to be changed accordingly

'''

from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dropout, Flatten, concatenate, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
import load_batch_data
import sys
import os
import pickle
from json import load as loadjson
import numpy as np
import gc


def conv2d_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1), bias=False):
    """
    Utility function to apply conv + BN.
    (Slightly modified from https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py)
    """
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1
    x = Conv2D(nb_filter, (nb_row, nb_col),
               strides=subsample,
               padding=border_mode,
               use_bias=bias)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x


def block_inception_a(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 96, 1, 1)

    branch_1 = conv2d_bn(input, 64, 1, 1)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3)

    branch_2 = conv2d_bn(input, 64, 1, 1)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)
    branch_2 = conv2d_bn(branch_2, 96, 3, 3)

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch_3 = conv2d_bn(branch_3, 96, 1, 1)

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    return x


def block_reduction_a(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 384, 3, 3, subsample=(2, 2), border_mode='valid')

    branch_1 = conv2d_bn(input, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 3, 3)
    branch_1 = conv2d_bn(branch_1, 256, 3, 3, subsample=(2, 2), border_mode='valid')

    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

    x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
    return x


def block_inception_b(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 384, 1, 1)

    branch_1 = conv2d_bn(input, 192, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 1, 7)
    branch_1 = conv2d_bn(branch_1, 256, 7, 1)

    branch_2 = conv2d_bn(input, 192, 1, 1)
    branch_2 = conv2d_bn(branch_2, 192, 7, 1)
    branch_2 = conv2d_bn(branch_2, 224, 1, 7)
    branch_2 = conv2d_bn(branch_2, 224, 7, 1)
    branch_2 = conv2d_bn(branch_2, 256, 1, 7)

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch_3 = conv2d_bn(branch_3, 128, 1, 1)

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    return x


def block_reduction_b(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 192, 1, 1)
    branch_0 = conv2d_bn(branch_0, 192, 3, 3, subsample=(2, 2), border_mode='valid')

    branch_1 = conv2d_bn(input, 256, 1, 1)
    branch_1 = conv2d_bn(branch_1, 256, 1, 7)
    branch_1 = conv2d_bn(branch_1, 320, 7, 1)
    branch_1 = conv2d_bn(branch_1, 320, 3, 3, subsample=(2, 2), border_mode='valid')

    branch_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(input)

    x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
    return x


def block_inception_c(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    branch_0 = conv2d_bn(input, 256, 1, 1)

    branch_1 = conv2d_bn(input, 384, 1, 1)
    branch_10 = conv2d_bn(branch_1, 256, 1, 3)
    branch_11 = conv2d_bn(branch_1, 256, 3, 1)
    branch_1 = concatenate([branch_10, branch_11], axis=channel_axis)

    branch_2 = conv2d_bn(input, 384, 1, 1)
    branch_2 = conv2d_bn(branch_2, 448, 3, 1)
    branch_2 = conv2d_bn(branch_2, 512, 1, 3)
    branch_20 = conv2d_bn(branch_2, 256, 1, 3)
    branch_21 = conv2d_bn(branch_2, 256, 3, 1)
    branch_2 = concatenate([branch_20, branch_21], axis=channel_axis)

    branch_3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(input)
    branch_3 = conv2d_bn(branch_3, 256, 1, 1)

    x = concatenate([branch_0, branch_1, branch_2, branch_3], axis=channel_axis)
    return x


def inception_v4_base(input):
    if K.image_dim_ordering() == "th":
        channel_axis = 1
    else:
        channel_axis = -1

    # Input Shape is 299 x 299 x 3 (th) or 3 x 299 x 299 (th)
    net = conv2d_bn(input, 32, 3, 3, subsample=(2, 2), border_mode='valid')
    net = conv2d_bn(net, 32, 3, 3, border_mode='valid')
    net = conv2d_bn(net, 64, 3, 3)

    branch_0 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(net)

    branch_1 = conv2d_bn(net, 96, 3, 3, subsample=(2, 2), border_mode='valid')

    net = concatenate([branch_0, branch_1], axis=channel_axis)

    branch_0 = conv2d_bn(net, 64, 1, 1)
    branch_0 = conv2d_bn(branch_0, 96, 3, 3, border_mode='valid')

    branch_1 = conv2d_bn(net, 64, 1, 1)
    branch_1 = conv2d_bn(branch_1, 64, 1, 7)
    branch_1 = conv2d_bn(branch_1, 64, 7, 1)
    branch_1 = conv2d_bn(branch_1, 96, 3, 3, border_mode='valid')

    net = concatenate([branch_0, branch_1], axis=channel_axis)

    branch_0 = conv2d_bn(net, 192, 3, 3, subsample=(2, 2), border_mode='valid')
    branch_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(net)

    net = concatenate([branch_0, branch_1], axis=channel_axis)

    # 35 x 35 x 384
    # 4 x Inception-A blocks
    for idx in xrange(4):
        net = block_inception_a(net)

    # 35 x 35 x 384
    # Reduction-A block
    net = block_reduction_a(net)

    # 17 x 17 x 1024
    # 7 x Inception-B blocks
    for idx in xrange(7):
        net = block_inception_b(net)

    # 17 x 17 x 1024
    # Reduction-B block
    net = block_reduction_b(net)

    # 8 x 8 x 1536
    # 3 x Inception-C blocks
    for idx in xrange(3):
        net = block_inception_c(net)

    return net


def inception_v4_model(img_rows, img_cols, color_type=1, num_classeses=None, dropout_keep_prob=0.2):
    '''
    Inception V4 Model for Keras

    Model Schema is based on
    https://github.com/kentsommer/keras-inceptionV4

    ImageNet Pretrained Weights
    Theano: https://github.com/kentsommer/keras-inceptionV4/releases/download/2.0/inception-v4_weights_th_dim_ordering_th_kernels.h5
    TensorFlow: https://github.com/kentsommer/keras-inceptionV4/releases/download/2.0/inception-v4_weights_tf_dim_ordering_tf_kernels.h5

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of class labels for our classification task
    '''

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    if K.image_dim_ordering() == 'th':
        inputs = Input((3, 299, 299))
    else:
        inputs = Input((299, 299, 3))

    # Make inception base
    net = inception_v4_base(inputs)

    # Final pooling and prediction

    # 8 x 8 x 1536
    net_old = AveragePooling2D((8, 8), padding='valid')(net)

    # 1 x 1 x 1536
    net_old = Dropout(dropout_keep_prob)(net_old)
    net_old = Flatten()(net_old)

    # 1536
    predictions = Dense(units=1001, activation='softmax')(net_old)

    model = Model(inputs, predictions, name='inception_v4')

    if K.image_dim_ordering() == 'th':
        # Use pre-trained weights for Theano backend
        weights_path = 'pretrained_models/inception-v4_weights_th_dim_ordering_th_kernels.h5'
    else:
        # Use pre-trained weights for Tensorflow backend
        weights_path = 'pretrained_models/inception-v4_weights_tf_dim_ordering_tf_kernels.h5'

    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    net_ft = AveragePooling2D((8, 8), padding='valid')(net)
    net_ft = Dropout(dropout_keep_prob)(net_ft)
    net_ft = Flatten()(net_ft)
    predictions_ft = Dense(units=num_classes, activation='softmax')(net_ft)

    model = Model(inputs, predictions_ft, name='inception_v4')

    # Learning rate is changed to 0.001
    sgd = SGD(lr=2e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    channel = 3
    batch_size = 16
    nb_epoch = 30
    img_rows, img_cols = 299, 299

    setname = sys.argv[1]
    threshold = sys.argv[2]
    step = sys.argv[3]
    if len(sys.argv) <= 4:
        root_dir = "/media/intel/m2/train_test_data"
    else:
        root_dir = sys.argv[4]
    tr_test_path = os.path.join(root_dir, step, setname, threshold, 'labels')
    with open(os.path.join(tr_test_path, 'class_name.json')) as fp:
        class_name = loadjson(fp)

    (num_classes, train_size, test_size, train_batches_count, test_batches_count, Y_test
     ) = load_batch_data.statistics_precal(tr_test_path, batch_size=batch_size)

    train_generator = load_batch_data.load_data(tr_test_path, target_size=(img_rows, img_cols),
                                                data_type='train', num_classes=num_classes,
                                                batch_size=batch_size, shuffle=True)

    # Load pre-trained model if there is one
    model = inception_v4_model(img_rows, img_cols, channel, num_classes, dropout_keep_prob=0.2)

    # Start Fine-tuning
    model.fit_generator(generator=train_generator,
                        epochs=nb_epoch,
                        max_queue_size=1,
                        workers=1,
                        verbose=2,
                        steps_per_epoch=train_batches_count
                        )
    gc.collect()

    # Make predictions
    test_generator = load_batch_data.load_data(tr_test_path, target_size=(img_rows, img_cols),
                                               data_type='test', num_classes=num_classes,
                                               batch_size=batch_size, shuffle=False)
    predictions_test = model.predict_generator(generator=test_generator,
                                               steps=test_batches_count,
                                               max_queue_size=1,
                                               workers=1,
                                               verbose=2
                                               )
    gc.collect()

    # Score
    final_pred = {}
    for i in xrange(len(Y_test)):
        img_class = list(Y_test[i]).index(1.)
        if img_class not in final_pred:
            final_pred[img_class] = []
        final_pred[img_class].append(list(predictions_test[i]))

    pred_save_path = os.path.join(root_dir, step, setname, threshold, 'final_pred', 'inceptionV4')
    if not os.path.isdir(pred_save_path):
        os.makedirs(pred_save_path)
    with open(os.path.join(pred_save_path, 'prob.pkl'), 'wb') as fp:
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
