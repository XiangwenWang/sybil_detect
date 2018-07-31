from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.utils import np_utils
import pandas as pd
import numpy as np
import os
import sys


def load_img_all(filenames, target_size=None, load_type=None):
    img_data, i = [], 0
    _file_count_limit = 512
    load_start = True
    img_data_all = None
    for filename in filenames:
        img = image.load_img(filename, target_size=target_size)
        img_data.append(image.img_to_array(img))
        img.close()
        i += 1
        if i % _file_count_limit == 0:
            print(i)
            sys.stdout.flush()
            img_data = np.array(img_data)
            if load_start:
                img_data_all = img_data
                load_start = False
            else:
                img_data_all = np.concatenate((img_data_all, img_data), axis=0)
            img_data = []
    if img_data:
        img_data = np.array(img_data)
        if load_start:
            img_data_all = img_data
        else:
            img_data_all = np.concatenate((img_data_all, img_data), axis=0)
    img_data_all = imagenet_utils.preprocess_input(img_data_all)
    # print('Finish loading %s data' % load_type)
    return img_data_all


def get_batch_counts(sample_count, batch_size):
    batches = int(sample_count / batch_size)
    if sample_count % batch_size != 0:
        batches += 1
    return batches


def load_data(foldername, target_size=None, data_type=None,
              re_gen_array=False, num_classes=None,
              batch_size=None, shuffle=False):

    df_file = os.path.join(foldername, "%s.txt" % data_type)
    df = pd.read_csv(df_file, sep=' ', header=None,
                     names=['filename', 'seller'])

    while True:
        if shuffle:
            df = df.sample(frac=1).reset_index(drop=True)

        batches = get_batch_counts(len(df), batch_size)

        for batch_index in xrange(0, batches):
            if (batch_index < batches - 1):
                batch_df = df.iloc[batch_index * batch_size:
                                   (batch_index + 1) * batch_size,
                                   :]
            else:
                batch_df = df.iloc[batch_index * batch_size:, :]

            y = np_utils.to_categorical(batch_df.seller.astype(int).values, num_classes)
            x = load_img_all(batch_df.filename.values, target_size=target_size)

            yield (x, y)


def statistics_precal(foldername, batch_size=None):
    train_file = os.path.join(foldername, 'train.txt')
    test_file = os.path.join(foldername, 'test.txt')
    df_dict = {}
    df_dict['train'] = pd.read_csv(train_file, sep=' ', header=None,
                                   names=['filename', 'seller'])
    df_dict['test'] = pd.read_csv(test_file, sep=' ', header=None,
                                  names=['filename', 'seller'])

    # train_ct, val_ct, test_ct = len(train_df), len(val_df), len(test_df)
    num_classes = len(pd.concat((df_dict['train'].seller, df_dict['test'].seller)).unique())

    train_batches_count = get_batch_counts(len(df_dict['train']), batch_size)
    test_batches_count = get_batch_counts(len(df_dict['test']), batch_size)

    test_Y = np_utils.to_categorical(df_dict['test'].seller.astype(int).values,
                                     num_classes)

    return (num_classes, len(df_dict['train']), len(df_dict['test']),
            train_batches_count, test_batches_count, test_Y)
