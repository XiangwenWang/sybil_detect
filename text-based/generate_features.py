# -*- coding: utf-8 -*-
'''
Requirement:
This code uses Stanford Log-linear Part-Of-Speech Tagger from
https://nlp.stanford.edu/software/tagger.shtml

Please download it and change the directory in tagger_pos accordingly
'''


import os
import re
from shutil import rmtree
from nltk.corpus import stopwords as sw
from nltk.tag import StanfordPOSTagger
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import multiprocessing
import sys
from __future__ import print_function


stop_words_list = sw.words('english')
punctus = ':-.,?;!\'\"'
startwithupper_r = re.compile(r"[A-Z].*")
containupper_r = re.compile(r".*[A-Z].*")

java_path = r"/home/tmpuser/jre/bin/java"
os.environ['JAVAHOME'] = java_path
os.environ['JAVA_HOME'] = java_path
tagger_pos = StanfordPOSTagger(r'/home/tmpuser/stanford-postagger-2017-06-09/' +
                               r'models/english-bidirectional-distsim.tagger',
                               r'/home/tmpuser/stanford-postagger-2017-06-09/' +
                               r'stanford-postagger.jar', java_options='-mx90000m')
pos_tag_list = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN',
                'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP',
                'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$',
                'WRB', '#', '$', "''", '(', ')', ',', '.', ':', '``', 'e', 'di', 'eff',
                'anti-infl', 'classifi', '-']
pos_tag_dict = {key: value for value, key in enumerate(pos_tag_list)}
pos_tag_len = len(pos_tag_list)

root_dir = 'description/%s/' % sys.argv[1]
feature_folder = 'stylometry_features/%s/' % sys.argv[1]

seller_name = sorted(os.listdir(root_dir))
seller_names = [os.path.join(root_dir, x) for x in seller_name]
'''
try:
    rmtree(feature_folder)
except OSError:
    pass
'''
try:
    os.makedirs(feature_folder)
except OSError:
    pass


def extract_feature(content, normed=False):

    feature_index = 0
    total_feature_count = (1 +  # word percentage that starts with upper letter
                           1 +  # word percentage what contains uppper letter
                           1 +  # upper letter percentage
                           1 +  # average word length
                           100 +  # word length
                           len(punctus) +  # punctuation
                           len(stop_words_list) +  # stop words
                           len(pos_tag_list) +  # pos tag unigram
                           len(pos_tag_list) ** 2 +  # pos tag bigram
                           len(pos_tag_list) ** 3 +  # pos tag trigram
                           95 +  # char unigram
                           95 ** 2 +  # char bigram
                           95 ** 3 +  # char trigram
                           10 +  # digit unigram
                           100 +  # digit bigram
                           1000 +  # digit trigram
                           0)
    feature_array = [0] * total_feature_count

    words = content.split(' ')
    words_total_count = float(len(words))
    startwithupper = len(filter(startwithupper_r.match, words)
                         ) / words_total_count if words_total_count else 0
    containupper = len(filter(containupper_r.match, words)
                       ) / words_total_count if words_total_count else 0
    feature_array[feature_index] = startwithupper
    feature_index += 1
    feature_array[feature_index] = containupper
    feature_index += 1

    char_total_count = float(len(re.findall(r'[A-z]', content)))
    char_upper_percent = len(re.findall(r'[A-Z]', content)
                             ) / char_total_count if char_total_count else 0
    feature_array[feature_index] = char_upper_percent
    feature_index += 1

    lengths = [len(w) for w in words]  # 100 features
    aver_word_len = float(sum(lengths)) / len(lengths) if lengths else 0
    feature_array[feature_index] = aver_word_len
    feature_index += 1
    lengths = [x for x in lengths if x <= 100 and x > 0]
    lengths_count = float(len(lengths)) if normed else 1
    lengths = Counter(lengths)
    for x in lengths:
        feature_array[feature_index + x - 1] = lengths[x] / lengths_count
    feature_index += 100

    punctuation = [x for x in content if x in punctus]
    punctuation_count = float(len(punctuation)) if normed else 1
    punctuation = Counter(punctuation)
    for x in punctuation:
        feature_array[feature_index + punctus.index(x)
                      ] = punctuation[x] / punctuation_count
    feature_index += len(punctus)

    stopwords = [x for x in words if x in stop_words_list]
    stopwords_count = float(len(stopwords)) if normed else 1
    stopwords = Counter(stopwords)
    for x in stopwords:
        feature_array[feature_index + stop_words_list.index(x)
                      ] = stopwords[x] / stopwords_count
    feature_index += len(stop_words_list)

    pos_tags = map(lambda x: [y[1] for y in x],
                   tagger_pos.tag_sents(map(word_tokenize, sent_tokenize(content))))

    pos_unigram = (sent_pos[i] for sent_pos in pos_tags
                   for i in xrange(len(sent_pos)))
    pos_unigram = Counter(pos_unigram)
    pos_unigram_count = float(sum(pos_unigram.values())) if normed else 1
    for x in pos_unigram:
        feature_array[feature_index + pos_tag_dict[x]
                      ] = pos_unigram[x] / pos_unigram_count
    feature_index += pos_tag_len

    pos_bigram = (tuple(sent_pos[i: i + 2]) for sent_pos in pos_tags
                  for i in xrange(len(sent_pos) - 1))
    pos_bigram = Counter(pos_bigram)
    pos_bigram_count = float(sum(pos_unigram.values())) if normed else 1
    for x in pos_bigram:
        feature_array[feature_index + pos_tag_dict[x[0]] * pos_tag_len +
                      pos_tag_dict[x[1]]] = pos_bigram[x] / pos_bigram_count
    feature_index += pos_tag_len ** 2

    pos_trigram = (tuple(sent_pos[i: i + 3]) for sent_pos in pos_tags
                   for i in xrange(len(sent_pos) - 2))
    pos_trigram = Counter(pos_trigram)
    pos_trigram_count = float(sum(pos_trigram.values())) if normed else 1
    for x in pos_trigram:
        feature_array[feature_index + pos_tag_dict[x[0]] * pos_tag_len ** 2 +
                      pos_tag_dict[x[1]] * pos_tag_len +
                      pos_tag_dict[x[2]]] = pos_trigram[x] / pos_trigram_count
    feature_index += pos_tag_len ** 3

    char_unigram = re.findall(r'(?=([ -~]{1}))', content)
    char_unigram_count = float(len(char_unigram)) if normed else 1
    char_unigram = Counter(char_unigram)
    for x in char_unigram:
        feature_array[feature_index + ord(x) - ord(' ')
                      ] = char_unigram[x] / char_unigram_count
    feature_index += 95

    char_bigram = re.findall(r'(?=([ -~]{2}))', content)
    char_bigram_count = float(len(char_bigram)) if normed else 1
    char_bigram = Counter(char_bigram)
    for x in char_bigram:
        feature_array[feature_index + (ord(x[0]) - ord(' ')) * 95 +
                      ord(x[1]) - ord(' ')] = char_bigram[x] / char_bigram_count
    feature_index += 95 ** 2

    char_trigram = re.findall(r'(?=([ -~]{3}))', content)
    char_trigram_count = float(len(char_trigram)) if normed else 1
    char_trigram = Counter(char_trigram)
    for x in char_trigram:
        feature_array[feature_index + (ord(x[0]) - ord(' ')) * 95 ** 2 +
                      (ord(x[1]) - ord(' ')) * 95 +
                      ord(x[2]) - ord(' ')] = char_trigram[x] / char_trigram_count
    feature_index += 95 ** 3

    dig_unigram = re.findall(r'(?=([0-9]{1}))', content)
    dig_unigram_count = float(len(dig_unigram)) if normed else 1
    dig_unigram = Counter(dig_unigram)
    for x in dig_unigram:
        feature_array[feature_index + int(x)] = dig_unigram[x] / dig_unigram_count
    feature_index += 10

    dig_bigram = re.findall(r'(?=([0-9]{2}))', content)
    dig_bigram_count = float(len(dig_bigram)) if normed else 1
    dig_bigram = Counter(dig_bigram)
    for x in dig_bigram:
        feature_array[feature_index + int(x)] = dig_bigram[x] / dig_bigram_count
    feature_index += 10 ** 2

    dig_trigram = re.findall(r'(?=([0-9]{3}))', content)
    dig_trigram_count = float(len(dig_trigram)) if normed else 1
    dig_trigram = Counter(dig_trigram)
    for x in dig_trigram:
        feature_array[feature_index + int(x)] = dig_trigram[x] / dig_trigram_count
    feature_index += 10 ** 3

    return feature_array


def get_feature(document_dstfolder):
    try:
        document = document_dstfolder[0]
        seller_feature_folder = document_dstfolder[1]
        doc_name = document.split('/')[-1]
        with open(document) as fp:
            content = fp.read()
            fp.close()
        X_raw = extract_feature(content, normed=False)
        with open(os.path.join(seller_feature_folder, doc_name), 'w') as fp:
            line = ''
            for j in xrange(len(X_raw)):
                x = X_raw[j]
                if not x:
                    line += '%d ' % int(x)
                elif abs(int(x) - x) < 1e-9:
                    line += '%d ' % int(x)
                else:
                    line += '%.6f ' % float(x)
            line = line[:-1]
            fp.write(line)
            fp.close()
    except:
        print(' ' + document)


mpp = multiprocessing.Pool(3)
for seller in seller_names:
    seller_index = seller_names.index(seller)
    seller_feature_folder = os.path.join(feature_folder, seller_name[seller_index])
    try:
        os.mkdir(seller_feature_folder)
    except OSError:
        pass
    documents = [os.path.join(seller, x) for x in sorted(os.listdir(seller))]
    multi_docu_dst = [(x, seller_feature_folder) for x in documents]
    if len(os.listdir(seller_feature_folder)) != len(documents):
        mpp.map(get_feature, multi_docu_dst)
    print(seller.split('/')[-1], end=' ')
