
import numpy as np
#import matplotlib as pt
import pandas as pd
from sklearn.model_selection import KFold #Import K-Fold validation from SKlearn
from sklearn.linear_model import LogisticRegression
import re, string
from sklearn.feature_extraction.text import TfidfVectorizer

# Remove any HTML tags from each text
def cleanhtml(raw_html): # Remove HTML tags 
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

# Remove special characters from each text
def remove_special_char(s):
    stripped = re.sub('^\s+|\s+', ' ', s) # large 
    stripped = re.sub('[^A-Za-z0-9 \']+', '', stripped)
    stripped = re.sub(r'^https?:\/\/.*[\r\n]*', '', stripped, flags=re.MULTILINE)
    stripped = stripped.strip()
    stripped = re.sub("\d+", "", stripped) # remove numbers
    return stripped

rawx_train = [] #Declare training X data
Y_train= [] #Declare label data
rawx_test = [] #Declare test set X data

rawdata = np.array(pd.read_csv('training.csv',index_col = None, header = None))
for i in range(len(rawdata)):
    clean = remove_special_char(cleanhtml(rawdata[i,0])).lower() 
    rawx_train.append(clean)
    Y_train.append(int(rawdata[i,1]))

Y_train = np.array(Y_train).reshape(-1,1)
      
testdata = np.array(pd.read_csv('test.csv',index_col = None, header = None))
for i in range(len(testdata)):
    clean = remove_special_char(cleanhtml(testdata[i,0])).lower() 
    rawx_test.append(clean)

def binary_feature():
	vecc = CountVectorizer(stop_words='english',ngram_range=(1, 1),binary = True)
	X_train = vecc.fit_transform(rawx_train)
	X_test = vecc.fit_transform(rawx_test)
	return X_train,X_test

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()	

def Tfidf(ng):
    vec = TfidfVectorizer(ngram_range=(1,ng), tokenizer=tokenize,
        strip_accents='unicode', use_idf=1,
       smooth_idf=1, sublinear_tf=1 )
    X_train = vec.fit_transform(rawx_train)
    X_test = vec.transform(rawx_test)
    return X_train,X_test
	
def BNB(trainx,trainy,valx):
    theta1 = np.sum(trainy)/len(trainy)
    yis01,yis00 = np.zeros((trainx.shape[1],), dtype=float),np.zeros((trainx.shape[1],), dtype=float)
    yis11,yis10 = np.zeros((trainx.shape[1],), dtype=float),np.zeros((trainx.shape[1],), dtype=float)
    for v in range(trainx.shape[1]):
        yis11[v] = (sum(1 for i in range(trainx.shape[0]) if trainy[i] == 1 and trainx[i,v] == 1)+1)/\
        (np.sum(trainy)+2) #Laplace smoothing applied
        yis10[v] = (sum(1 for i in range(trainx.shape[0]) if trainy[i] == 0 and trainx[i,v] == 1)+1)/\
        (trainy.shape[0]-np.sum(trainy)+2) #Laplace smoothing applied
        yis00[v] = (sum(1 for i in range(trainx.shape[0]) if trainy[i] == 0 and trainx[i,v] == 0)+1)/\
        (trainy.shape[0]-np.sum(trainy)+2) #Laplace smoothing applied
        yis01[v] = (sum(1 for i in range(trainx.shape[0]) if trainy[i] == 1 and trainx[i,v] == 0)+1)/\
        (np.sum(trainy)+2) #Laplace smoothing applied
    linpart = (1-valx)@np.log(yis01/yis00)+valx@np.log(yis11/yis10);
    probs = np.tile(np.log(theta1/(1-theta1)),[valx.shape[0],1])+linpart.reshape(len(linpart),1)
    predicty = np.asarray([1 if k > 0 else 0 for k in probs])
    return predicty

#X_train,X_test = binary_feature()

def test(penal,ng,cc):
    X_train,X_test = Tfidf(ng)
    fold = 5
    kf = KFold(n_splits = fold)
    #accuracy_BNB,accuracy_LR,accuracy_SVM = np.zeros(fold),np.zeros(fold),np.zeros(fold),np.zeros(fold)
    accuracy_LR = np.zeros(fold)
    count = 0
    
    for train_index, val_index in kf.split(X_train):
        trainx, valx = X_train[train_index,:], X_train[val_index,:]
        trainy, valy = Y_train[train_index],Y_train[val_index]
        
        #run support vector machine from scikit
        #supportv = svm.SVC(gamma=0.01)
        #supportv.fit(trainx, trainy) 
        #accuracy_SVM[count] = accuracy_score(valy, svm.predict(valx))
        
        #run naive bayes method
        #predicty = BNB(trainx,trainy,valx)
        #accuracy_BNB[count] = sum(1 for i in range(valx.shape[0]) if valy[i] == predicty[i])*100/valx.shape[0]
    
        #run logistic regression from scikit
        lg = LogisticRegression(random_state=0,max_iter=10000,penalty=penal,C=cc).fit(trainx, np.ravel(trainy))
        accuracy_LR[count] = lg.score(valx, valy)*100
     
        #print(count,' fold has been completed.')
        count += 1
    
    
    #rint('Average K-fold validation prediction accuracy using Bernoulli Naive Bayes is: ',np.average(accuracy_BNB),'%')
    print('When ngram is',ng,'and penalty is',penal)
    print('Average K-fold validation prediction accuracy using Logistic Regression is: ',np.average(accuracy_LR),'%')
    #print('Average K-fold validation prediction accuracy using Support Vector Machine is: ',np.average(accuracy_SVM),'%')
    return np.average(accuracy_LR)

bdict = ['l1','l2']
adict = [1,2,3]
cdict = [0.1,1,10]

result = np.zeros((2,3,3))
for c in range(len(cdict)):    
    for a in range(len(adict)):
        for b in range(len(bdict)):
            result[b,a,c] = test(bdict[b],adict[a],cdict[c])
