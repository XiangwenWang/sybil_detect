# -*- coding: utf-8 -*-
'''
Code for ground truth evaluation of sybil detection with photos with Residual Network model

The implementation of the Residual-50 model is based on
https://github.com/flyyufelix/cnn_finetune/blob/master/resnet_50.py

* For ground truth evaluation, large vendors will be split into two pesudo vendors, which will be
added into training and testing set respectively
* Network model was pre-trained on ImageNet, and training data are used to finetune the network weights
* Prediction is made on each image in testing set and results are averaged to obtain the
vendor similarities

To use the code, the data folder need to be changed accordingly
'''


from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import ZeroPadding2D, Flatten, add, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras import backend as K
import load_batch_data
import sys
import os
import pickle
from json import load as loadjson
import numpy as np
import gc


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    The identity_block is the block that has no conv layer at shortcut
    Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size),
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """

    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet50_model(img_rows, img_cols, color_type=1, num_classes=None):
    """
    Resnet 50 Model for Keras

    Model Schema is based on
    https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py

    ImageNet Pretrained Weights
    https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of class labels for our classification task
    """

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(shape=(img_rows, img_cols, color_type))
    else:
        bn_axis = 1
        img_input = Input(shape=(color_type, img_rows, img_cols))

    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    # Fully Connected Softmax Layer
    x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_fc = Flatten()(x_fc)
    x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)

    # Create model
    model = Model(img_input, x_fc)

    # Load ImageNet pre-trained data
    if K.image_dim_ordering() == 'th':
        # Use pre-trained weights for Theano backend
        weights_path = 'pretrained_models/resnet50_weights_th_dim_ordering_th_kernels.h5'
    else:
        # Use pre-trained weights for Tensorflow backend
        weights_path = 'pretrained_models/resnet50_weights_tf_dim_ordering_tf_kernels.h5'

    model.load_weights(weights_path)

    # Truncate and replace softmax layer for transfer learning
    # Cannot use model.layers.pop() since model is not of Sequential() type
    # The method below works since pre-trained weights are stored in layers but not in the model
    x_newfc = AveragePooling2D((7, 7), name='avg_pool')(x)
    x_newfc = Flatten()(x_newfc)
    x_newfc = Dense(num_classes, activation='softmax', name='fc10')(x_newfc)

    # Create another model with our customized softmax
    model = Model(img_input, x_newfc)

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == '__main__':
    channel = 3
    batch_size = 16
    nb_epoch = 15
    img_rows, img_cols = 224, 224

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
    finetuned_model = os.path.join(finetuned_models_folder, "resnet50_%s.h5" % threshold)
    if os.path.isfile(finetuned_model):
        model = load_model(finetuned_model)
        print('Loaded fine-tuned model from %s' % finetuned_model)
    else:
    	# Otherwise, load pre-trained model

        model = resnet50_model(img_rows, img_cols, channel, num_classes)

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

    pred_save_path = os.path.join(root_dir, step, setname, threshold, 'final_pred', 'Resnet50')
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
