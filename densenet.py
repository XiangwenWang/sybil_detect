# -*- coding: utf-8 -*-
'''
Code for ground truth evaluation of sybil detection with photos with DenseNet model

The implementation of the DenseNet-121 model is based on
https://github.com/flyyufelix/cnn_finetune/blob/master/densenet121.py

* For ground truth evaluation, large vendors will be split into two pesudo vendors, which will be
added into training and testing set respectively
* Network model was pre-trained on ImageNet, and training data are used to finetune the network weights
* Prediction is made on each image in testing set and results are averaged to obtain the
vendor similarities

To use the code, the data folder need to be changed accordingly
'''


from keras.optimizers import SGD
from keras.layers import Input, concatenate, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
import keras.backend as K
from misc_files.scale_layer import Scale
import load_batch_data
import sys
import os
import pickle
from json import load as loadjson
import numpy as np
import gc

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def densenet121_model(img_rows, img_cols, color_type=1, nb_dense_block=4,
                      growth_rate=32, nb_filter=64, reduction=0.5,
                      dropout_rate=0.0, weight_decay=1e-4, num_classes=None):
    '''
    DenseNet 121 Model for Keras

    Model Schema is based on
    https://github.com/flyyufelix/DenseNet-Keras

    ImageNet Pretrained Weights
    Theano: https://drive.google.com/open?id=0Byy2AcGyEVxfMlRYb3YzV210VzQ
    TensorFlow: https://drive.google.com/open?id=0Byy2AcGyEVxfSTA4SHJVOHNuTXc

    # Arguments
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters
        reduction: reduction factor of transition blocks.
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        classes: optional number of classes to classify images
        weights_path: path to pre-trained weights
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5

    # compute compression factor
    compression = 1.0 - reduction

    # Handle Dimension Ordering for different backends
    global concat_axis
    if K.image_dim_ordering() == 'tf':
        concat_axis = 3
        img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
    else:
        concat_axis = 1
        img_input = Input(shape=(color_type, img_rows, img_cols), name='data')

    # From architecture for ImageNet (Table 1 in the paper)
    nb_filter = 64
    nb_layers = [6, 12, 24, 16]  # For DenseNet-121

    # Initial convolution
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(nb_filter, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv1_bn')(x)
    x = Scale(axis=concat_axis, name='conv1_scale')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx + 2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name='conv' + str(final_stage) + '_blk_bn')(x)
    x = Scale(axis=concat_axis, name='conv' + str(final_stage) + '_blk_scale')(x)
    x = Activation('relu', name='relu' + str(final_stage) + '_blk')(x)

    x_fc = GlobalAveragePooling2D(name='pool' + str(final_stage))(x)
    x_fc = Dense(1000, name='fc6')(x_fc)
    x_fc = Activation('softmax', name='prob')(x_fc)

    model = Model(img_input, x_fc, name='densenet')

    if K.image_dim_ordering() == 'th':
        # Use pre-trained weights for Theano backend
        weights_path = 'pretrained_models/densenet121_weights_th.h5'
    else:
        # Use pre-trained weights for Tensorflow backend
        weights_path = 'pretrained_models/densenet121_weights_tf.h5'

    model.load_weights(weights_path, by_name=True)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = GlobalAveragePooling2D(name='pool' + str(final_stage))(x)
    x_newfc = Dense(num_classes, name='fc6')(x_newfc)
    x_newfc = Activation('softmax', name='prob')(x_newfc)

    model = Model(img_input, x_newfc)

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def conv_block(x, stage, branch, nb_filter, dropout_rate=None, weight_decay=1e-4):
    '''Apply BatchNorm, Relu, bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            branch: layer index within each dense block
            nb_filter: number of filters
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''
    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x1_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_x1_scale')(x)
    x = Activation('relu', name=relu_name_base + '_x1')(x)
    x = Conv2D(inter_channel, (1, 1), name=conv_name_base + '_x1', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_x2_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_x2_scale')(x)
    x = Activation('relu', name=relu_name_base + '_x2')(x)
    x = ZeroPadding2D((1, 1), name=conv_name_base + '_x2_zeropadding')(x)
    x = Conv2D(nb_filter, (3, 3), name=conv_name_base + '_x2', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    '''
    Apply BatchNorm, 1x1 Convolution, averagePooling, optional compression, dropout
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_filter: number of filters
            compression: calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
            dropout_rate: dropout rate
            weight_decay: weight decay factor
    '''

    eps = 1.1e-5
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage)

    x = BatchNormalization(epsilon=eps, axis=concat_axis, name=conv_name_base + '_bn')(x)
    x = Scale(axis=concat_axis, name=conv_name_base + '_scale')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Conv2D(int(nb_filter * compression), (1, 1), name=conv_name_base, use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2), name=pool_name_base)(x)

    return x


def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4, grow_nb_filters=True):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x: input tensor
            stage: index for dense block
            nb_layers: the number of layers of conv_block to append to the model.
            nb_filter: number of filters
            growth_rate: growth rate
            dropout_rate: dropout rate
            weight_decay: weight decay factor
            grow_nb_filters: flag to decide to allow number of filters to grow
    '''

    # eps = 1.1e-5
    concat_feat = x

    for i in range(nb_layers):
        branch = i + 1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate, weight_decay)
        concat_feat = concatenate([concat_feat, x], axis=concat_axis,
                                  name='concat_' + str(stage) + '_' + str(branch))

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter


if __name__ == '__main__':

    channel = 3
    batch_size = 16
    nb_epoch = 15
    img_rows, img_cols = 224, 224  # Resolution of inputs

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

    # Load fine-tuned model if there is one

    finetuned_models_folder = os.path.join(root_dir, step, 'finetuned_models', setname)
    finetuned_model = os.path.join(finetuned_models_folder, "densenet121_%s.h5" % threshold)
    if os.path.isfile(finetuned_model):
        model = load_model(finetuned_model)
        print('Loaded fine-tuned model from %s' % finetuned_model)
    else:

    	# Otherwise, load pre-trained model
        model = densenet121_model(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)

        # Start Fine-tuning
        model.fit_generator(generator=train_generator,
                            epochs=nb_epoch,
                            max_queue_size=1,
                            class_weight=None,
                            workers=1,
                            verbose=2,
                            steps_per_epoch=train_batches_count
                            )
        gc.collect()
        try:
            os.makedirs(finetuned_models_folder)
        except OSError:
            pass
        model.save(finetuned_model)

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

    pred_save_path = os.path.join(root_dir, step, setname, threshold, 'final_pred', 'densenet121')
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
