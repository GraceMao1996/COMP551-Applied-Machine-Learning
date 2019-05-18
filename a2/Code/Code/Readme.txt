To read raw txt files into .csv files, please run Read.py first. In this package, train and test.csv are already provided.

Required library for Read:
numpy
os
csv

After the train.csv and test.csv are generated, please place them in the same folder as the classifer code.

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

Required library for NBSVM:
numpy
pandas
re,string
sklearn.feature_extraction.text --TfidfVectorizer
sklearn.model_selection -- KFold
sklearn.metrics -- accuracy_score
sklearn.svm -- LinearSVC

Simply run NBSVM.py with train.csv and test.csv in the same folder. A excel file named "test_out_NBSVM_testagain.csv" will be generated ready for Kaggle submission.

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

Required library for LG:
numpy
pandas
re,string
sklearn.feature_extraction.text --TfidfVectorizer
sklearn.model_selection -- KFold
sklearn.linear_model -- LogisticRegression

Simply run Logistic Regression.py with train.csv and test.csv in the same folder. Result matrix records the performance of different training conditions.

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

Required library for BNB:
numpy
pandas
re
sklearn.feature_extraction.text --CountVectorizer
sklearn.model_selection -- KFold

Simply run BNB.py with train.csv and test.csv in the same folder.

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------

Required library for SVM:
numpy
pandas
re,string
sklearn.feature_extraction.text --TfidfVectorizer
sklearn.model_selection -- KFold
sklearn.linear_model -- LinearSVC

Simply run SVM.py with train.csv and test.csv in the same folder. Result matrix records the performance of different training conditions.