# %%
# gensim modules
import nltk

nltk.download('popular')
# %%
# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec

import csv
import re
import random
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
# classifier
from nltk.sentiment.util import mark_negation as mkneg
import logging
import sys

log = logging.getLogger()
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)


# %%


def cleanhtml(raw_html):  # Remove HTML tags
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


trainingset = []  # each input
trainingScore = []
trainingnumber = []
# Trianing.csv has two column  1st column is comment and 2nd column is 1/0 (pos/neg)
# This step split the data into two lists, trainingset for comment and trainingScore stores the pos/neg
with open('training.csv', newline='', encoding='Latin-1')   as readCSV:
    readCSV = csv.reader(readCSV)
    index = 0
    for row in readCSV:  # row[0] = comment ,row[1] = score
        if index in range(0, 25000):
            data = cleanhtml(row[0])
            data = data.lower().strip()
            data = data.replace('...', ' ')
            data = data.replace('.', '\t')  # replace '.' with '/t'
            data = data.replace('?', '\t')
            data = data.replace('!', '\t')
            for sentence in data.split('\t'):
                trainingset.append(sentence)
            trainingnumber.append(len(data.split('\t')))
            trainingScore.append(row[1])
            index += 1
        else:
            trainingset.append(row[0])
            trainingnumber.append(1)
            trainingScore.append(row[1])
print(len(trainingset))


# Tokenization and Stemmer and negation handling


# %%
#####################################################################
class TaggedLineSentence(object):
    def __init__(self, trainingset):
        self.trainingset = trainingset

    def __iter__(self):
        for i, row in enumerate(self.trainingset):
            yield TaggedDocument(word_tokenize(row), [i])

    def to_array(self):
        j = 1
        self.sentences = []
        for i, row in enumerate(self.trainingset):
            self.sentences.append(TaggedDocument(word_tokenize(row), [i]))
        return (self.sentences)

    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return (shuffled)


log.info('source load')
source = trainingset
log.info('TaggedDocument')
sentences = TaggedLineSentence(source)
sentences.to_array()
vector_size = 100
# %%
log.info('D2V')
model = Doc2Vec(min_count=1, window=10, vector_size=vector_size, sample=1e-3, workers=8, dm=0, dm_concat=0,
                dbow_words=0)
model.build_vocab(sentences.to_array())
log.info('Epoch')
model.train(sentences, total_examples=model.corpus_count, epochs=13)
log.info('Model Save')
model.save('./bagofword_400_win_5.d2v')
# %%

log.info('Sentiment')
train_arrays = np.zeros((len(trainingset), vector_size))
print(len(trainingset))
j = 0
for i in range(len(trainingset)):
    train_arrays[i] = model.docvecs[i]
with open('out.emb', 'w') as f:
    index = 0
    for i, number in enumerate(trainingnumber):
        train_each_group = train_arrays[index:index + number]
        index = index + number
        for embed in train_each_group:
            for number in embed:
                f.write(str(number))
                f.write(' ')
            f.write('\t')
        f.write('|')
        f.write(str(trainingScore[i]))
        f.write('\n')



