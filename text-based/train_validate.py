# -*- coding: utf-8 -*-
'''
The code consists of three parts:
1. Generate training and testing data
2. Dimension reduction
3. Logistic regression
'''


import os
import pandas as pd
from numpy.random import shuffle
from shutil import rmtree
from shutil import move
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD
import sys
from pickle import dump as savepkl


if sys.argv[3] == '1':
    prefix = 'nondup_'
else:
    prefix = ''
src_dir = '%sstylometry_features/%s/' % (prefix, sys.argv[1])
dst_dir = '%sstylometry_train_val/%s_%s/train' % (prefix, sys.argv[1], sys.argv[2])

try:
    rmtree(dst_dir)
except:
    pass
try:
    os.makedirs(dst_dir)
except:
    pass

root_dir = '%sdescription/%s/' % (prefix, sys.argv[1])
sellers = os.listdir(root_dir)

wc_seller = {}
for seller in sellers:
    seller_path = os.path.join(root_dir, seller)
    descps = [os.path.join(seller_path, x) for x in os.listdir(seller_path)]
    word_count = 0
    for des in descps:
        with open(des) as fp:
            content = fp.read()
            word_count += len(content.split(' '))
    wc_seller[seller] = word_count
df = pd.DataFrame.from_dict(wc_seller, orient='index')
df.columns = ['word_count']

_threshold = int(sys.argv[2])
train_seller = df[df.word_count >= _threshold].sort_index().index.values
val_seller = df[df.word_count >= _threshold * 2].sort_index().index.values

print('training seller:', len(train_seller), 'validation seller:', len(val_seller))

for i in xrange(len(train_seller)):
    src_seller = os.path.join(src_dir, train_seller[i])
    dst_seller = os.path.join(dst_dir, train_seller[i])
    try:
        os.rename(src_seller, dst_seller)
    except:
        pass
    '''
    os.makedirs(dst_seller)
    for doc in os.listdir(src_seller):
        copy2(os.path.join(src_seller, doc), os.path.join(dst_seller, doc))
    '''

src_dir = '%sstylometry_train_val/%s_%s/train' % (prefix, sys.argv[1], sys.argv[2])
dst_dir = '%sstylometry_train_val/%s_%s/val' % (prefix, sys.argv[1], sys.argv[2])

for i in xrange(len(val_seller)):
    src_seller = os.path.join(src_dir, val_seller[i])
    dst_seller = os.path.join(dst_dir, val_seller[i])
    os.makedirs(dst_seller)
    docs = os.listdir(src_seller)
    shuffle(docs)
    docs = docs[len(docs) / 2:]
    for doc in docs:
        src_doc = os.path.join(src_seller, doc)
        dst_doc = os.path.join(dst_seller, doc)
        move(src_doc, dst_doc)

X, y = None, []
Start = True
X_tmp, count_tmp = [], 0
for i in xrange(len(train_seller)):
    seller = os.path.join(src_dir, train_seller[i])
    for doc in os.listdir(seller):
        y.append(i)
        vec = pd.read_csv(os.path.join(seller, doc), header=None, lineterminator=' ').values.flatten()
        X_tmp.append(vec)
        count_tmp += 1
        if count_tmp % 500 == 0:
            X_tmp = csr_matrix(X_tmp)
            X = X_tmp if Start else vstack([X, X_tmp])
            Start = False
            X_tmp, count_tmp = [], 0
            print("  |  %d seller finished" % len(y))
        if count_tmp % 100 == 0:
            print(count_tmp),
X_tmp = csr_matrix(X_tmp)
X = X_tmp if Start else vstack([X, X_tmp])

y = np.array(y)
svd = TruncatedSVD(n_components=1000, n_iter=10)
X_new = svd.fit_transform(X)

train_seller_list = list(train_seller)
X_val, y_val = None, []
Start = True
X_tmp, count_tmp = [], 0
for i in xrange(len(val_seller)):
    seller = os.path.join(dst_dir, val_seller[i])
    seller_index = train_seller_list.index(val_seller[i])
    for doc in os.listdir(seller):
        y_val.append(seller_index)
        vec = pd.read_csv(os.path.join(seller, doc), header=None, lineterminator=' ').values.flatten()
        X_tmp.append(vec)
        count_tmp += 1
        if count_tmp % 500 == 0:
            X_tmp = csr_matrix(X_tmp)
            X_val = X_tmp if Start else vstack([X_val, X_tmp])
            Start = False
            X_tmp, count_tmp = [], 0
            print("  |  %d seller finished" % i)
        if count_tmp % 100 == 0:
            print(count_tmp),
X_tmp = csr_matrix(X_tmp)
X_val = X_tmp if Start else vstack([X_val, X_tmp])

X_val_new = svd.transform(X_val)

with open('%sstylometry_train_val/%s_%s/svd_1000_train_val.pkl' %
          (prefix, sys.argv[1], sys.argv[2]), 'wb') as fp:
    savepkl((X_new, y, X_val_new, y_val), fp)

clf = LogisticRegression(penalty='l1', n_jobs=12, solver='saga')

index_rnd = np.arange(len(X_new))
shuffle(index_rnd)

X_new2 = X_new[index_rnd]
y_new2 = y[index_rnd]

clf.fit(X_new2, y_new2)
res = clf.predict_proba(X_val_new)

y_pred = []
start, last, res_tmp = True, -1, None
for i in xrange(len(y_val)):
    if y_val[i] != last:
        if not start:
            y_pred.append([last, res_tmp.argmax()])
        else:
            start = False
        res_tmp = res[i].copy()
    else:
        res_tmp += res[i]
    last = y_val[i]
y_pred.append([last, res_tmp.argmax()])

wrong = [tr_pred for tr_pred in y_pred if tr_pred[0] != tr_pred[1]]

print('total:', len(y_pred),
      'wrong:', len(wrong),
      'accuracy: %.5f' % (1 - (len(y_pred) * 1. - len(wrong)) / len(y_pred)))

print(wrong)
