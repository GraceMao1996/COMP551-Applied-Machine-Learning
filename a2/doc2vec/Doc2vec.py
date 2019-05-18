#%%
# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec
from sklearn.naive_bayes import MultinomialNB
import csv
import re
import random
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
# classifier
from nltk.sentiment.util import mark_negation as mkneg
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import KFold #Import K-Fold validation from SKlearn
from sklearn import svm
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import logging
import sys

log = logging.getLogger()
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)
#%%


def cleanhtml(raw_html): # Remove HTML tags 
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext
# input string
def remove_special_char(s):
    stripped = re.sub('^\s+|\s+', ' ', s) # large 
    stripped = re.sub(r'^https?:\/\/.*[\r\n]*', '', stripped, flags=re.MULTILINE)
    stripped = re.sub('[^A-Za-z0-9 \']+', '', stripped)
    stripped = stripped.strip()
    # stripped = re.sub("\d+", "", stripped) # remove numbers
    return stripped

trainingset = [] # each input 
trainingScore = []
trainingnumber=[]
# Trianing.csv has two column  1st column is comment and 2nd column is 1/0 (pos/neg)
# This step split the data into two lists, trainingset for comment and trainingScore stores the pos/neg
with open('training.csv', newline = '',encoding='Latin-1' )   as readCSV:  
    readCSV = csv.reader(readCSV, delimiter='|')
    for row in readCSV: # row[0] = comment ,row[1] = score
        data = cleanhtml(row[0]) 

        data = remove_special_char(data).lower()
        data = data.replace('.','\t')   #replace '.' with '/t'
		data = data.replace('?','\t')
		data = data.replace('!','\t')
		for sentence in data.strip('\t'):
            trainingset.append(sentence)
			trainingnumber.append(len(data.strip('\t')))
            trainingScore.append(row[1])


 #Tokenization and Stemmer and negation handling



#%%
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
        return(self.sentences)
		
    def sentences_perm(self):
        shuffled = list(self.sentences)
        random.shuffle(shuffled)
        return(shuffled)



log.info('source load')
source = trainingset
log.info('TaggedDocument')
sentences = TaggedLineSentence(source)
sentences.to_array()
vector_size = 400
#%%
log.info('D2V')
model = Doc2Vec(min_count=1, window=10, vector_size=vector_size, sample=1e-5, workers=8, dm = 0, dm_concat = 1, dbow_words =1)
model.build_vocab(sentences.to_array())

log.info('Epoch')
for epoch in range(10):
	log.info('EPOCH: {}'.format(epoch))
	model.train(sentences.sentences_perm(),epochs=model.iter, total_examples=model.corpus_count)

log.info('Model Save')
model.save('./bagofword_400_win_5.d2v')
#%%
model = Doc2Vec.load('./bagofword_500.d2v')
log.info('Sentiment')
train_arrays = np.zeros((25000, vector_size))
train_labels = np.zeros(25000)

for i in range(12500):
    prefix_train_pos =  str(1)
    prefix_train_neg =  str(0)
    train_arrays[i] = model.docvecs[i]
    train_arrays[12500 + i] = model.docvecs[12500 + i]
    train_labels[i] = 1
    train_labels[12500 + i] = 0

log.info(train_labels)
#%%
def LSVM(C_):
    log.info('K fold validation')
    fold = 5
    kf = KFold(n_splits = fold, shuffle= True)
    accuracy_LSVM = np.zeros(fold)


    count = 0
    X_train = train_arrays
    Y_train = train_labels
    # trainx : trainingset, valx : validationx
    # trainy : trainingset, valy: validationy
    for train_index, val_index in kf.split(X_train):
        print("count: "+ str(count))

        trainx, valx = X_train[train_index,:], X_train[val_index,:]
        trainy, valy = Y_train[train_index],Y_train[val_index]

        
        linearsvm = LinearSVC(C=C_, max_iter= 10000)
        linearsvm.fit(trainx, trainy)
        accuracy_LSVM[count] = linearsvm.score(valx,valy)*100

        # logistic classifier
        # clf = LogisticRegression(max_iter=4000, tol=5e-5, penalty= "l2")
        # clf.fit(trainx, trainy)


        

        count = count + 1
        print(linearsvm.score(valx, valy))

    print('Average K-fold validation prediction accuracy using Linear SVM is: ',np.average(accuracy_LSVM),'%')
def LG(C_):
    log.info('K fold validation')
    fold = 5
    kf = KFold(n_splits = fold, shuffle= True)
    accuracy_LG = np.zeros(fold)


    count = 0
    X_train = train_arrays
    Y_train = train_labels
    # trainx : trainingset, valx : validationx
    # trainy : trainingset, valy: validationy
    for train_index, val_index in kf.split(X_train):
        print("count: "+ str(count))

        trainx, valx = X_train[train_index,:], X_train[val_index,:]
        trainy, valy = Y_train[train_index],Y_train[val_index]

        
        # linearsvm = LinearSVC(C=C_, max_iter= 10000)
        # linearsvm.fit(trainx, trainy)
        # accuracy_LSVM[count] = linearsvm.score(valx,valy)*100

        # logistic classifier
        clf = LogisticRegression(max_iter=4000, tol=5e-5, penalty= "l2", C = C_)
        clf.fit(trainx, trainy)


        

        count = count + 1
        print(linearsvm.score(valx, valy))

    print('Average K-fold validation prediction accuracy using Logistic Regression is: ',np.average(accuracy_LSVM),'%')

#%%
LSVM(10)
