# -*- coding: utf-8 -*-
'''
Code for ground truth evaluation of sybil detection with photos with VGG model

The implementation of the VGG-19 model is based on
https://github.com/flyyufelix/cnn_finetune/blob/master/vgg19.py

* For ground truth evaluation, large vendors will be split into two pesudo vendors, which will be
added into training and testing set respectively
* Network model was pre-trained on ImageNet, and training data are used to finetune the network weights
* Prediction is made on each image in testing set and results are averaged to obtain the
vendor similarities

To use the code, the data folder need to be changed accordingly
'''


from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import ZeroPadding2D, Dropout, Flatten
import load_batch_data
import sys
import os
import pickle
from json import load as loadjson
import numpy as np
import gc

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def vgg19_model(img_rows, img_cols, channel=1, num_classes=None):
    """
    VGG 19 Model for Keras

    Model Schema is based on
    https://gist.github.com/baraldilorenzo/8d096f48a1be4a2d660d

    ImageNet Pretrained Weights
    https://drive.google.com/file/d/0Bz7KyqmuGsilZ2RVeVhKY0FyRmc/view?usp=sharing

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of class labels for our classification task
    """

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols, channel)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    # Loads ImageNet pre-trained data
    model.load_weights('pretrained_models/vgg19_weights_tf_dim_ordering_tf_kernels.h5')

    # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes, activation='softmax'))

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


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

    # Load pre-trained model
    model = vgg19_model(img_rows, img_cols, channel, num_classes)

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

    pred_save_path = os.path.join(root_dir, step, setname, threshold, 'final_pred', 'VGG19')
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
