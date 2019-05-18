# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:09:11 2019

@author: Max
"""
import numpy as np
import pandas as pd
import re, string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold #Import K-Fold validation from SKlearn
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

rawx_train = [] #Declare training X data
Y_train= [] #Declare label data
rawx_test = [] #Declare test set X data

rawdata = np.array(pd.read_csv('training.csv',index_col = None, header = None))
for i in range(len(rawdata)):
    clean = rawdata[i,0].lower()
    rawx_train.append(clean)
    Y_train.append(int(rawdata[i,1]))

Y_train = np.array(Y_train).reshape(-1,1)
      
testdata = np.array(pd.read_csv('test.csv',index_col = None, header = None))
for i in range(len(testdata)):
    clean = testdata[i,0].lower()
    rawx_test.append(clean)
    
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

def pr(alpha,x, y_i, y):
    p = x[y==y_i].sum(0)
    return (p+alpha) / ((y==y_i).sum()+alpha)

def get_mdl(alpha,x,y,const):
    r = np.log(pr(alpha,x,1,y) / pr(alpha,x,0,y))
    m = LinearSVC(max_iter = 100000,C = const)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r

def test(alpha,c,minf,maxf,beta):
    print('Training classifier...')
    
    vec = TfidfVectorizer(ngram_range=(1,3), tokenizer=tokenize,
       min_df=minf, max_df=maxf, strip_accents='unicode', use_idf=1,
       smooth_idf=1, sublinear_tf=1 )
    X_train = vec.fit_transform(rawx_train)
    X_test = vec.transform(rawx_test)
    fold = 5
    kf = KFold(n_splits = fold,shuffle=True)
    count = 0
    accuracy,avgr,avgw_prime,avg_svm_inter = np.zeros(fold),[],[],np.zeros(fold)
    
    for train_index, val_index in kf.split(X_train):
        trainx, valx = X_train[train_index,:], X_train[val_index,:]
        trainy, valy = Y_train[train_index],Y_train[val_index]
        
        trainy, valy = np.ravel(trainy),np.ravel(valy) #Flatten 
        
        m,r = get_mdl(alpha,trainx,trainy,c)
        avgr.append(r)
        svm_coef = m.coef_
        svm_interc = m.intercept_
        w_bar = (abs(svm_coef).sum())/trainx.shape[1]
        w_prime = (1 - beta)*(w_bar) + (beta * svm_coef)
        avgw_prime.append(w_prime)
        avg_svm_inter[count] = svm_interc
        
        part1 = valx.multiply(r)
        part2 = w_prime.T
        part3 = part1 @ part2 + svm_interc
        predy_sign = np.sign(part3)
        predy = [0 if i == -1 else 1 for i in predy_sign]
        accuracy[count] = accuracy_score(predy, valy)
        count += 1
        print(count,' fold has been completed.')
    
    r = np.mean(avgr,axis=0)
    w_prime = np.mean(avgw_prime,axis=0)
    svm_interc = np.mean(avg_svm_inter,axis=0)
    
    #test_out = m.predict(X_test.multiply(r))
    tpredy_sign = np.sign((X_test.multiply(r)) @ w_prime.T + svm_interc)
    test_out = [0 if i == -1 else 1 for i in tpredy_sign]
    index = range(0,len(test_out))
    dataframe = pd.DataFrame({'Id':index,'Category':np.ravel(test_out)})
    dataframe.to_csv("test_out_NBSVM_testagain.csv",index=False,sep=',')
    print('Average K-fold validation prediction accuracy using NBSVM is: ',np.average(accuracy)*100,'%')

test(0.1,10,3,0.9,0.5)