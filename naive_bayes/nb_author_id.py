#!/usr/bin/python

""" 
    this is the code to accompany the Lesson 1 (Naive Bayes) mini-project 

    use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1

"""
    
import sys
import os
from time import time
sys.path.append("/Users/selvaprabhakaran/Documents/work/pythonwork/python_ML_git/ud120-projects/tools/")
from email_preprocess import preprocess

# Set cwd
os.chdir("/Users/selvaprabhakaran/Documents/work/pythonwork/python_ML_git/ud120-projects/tools/")

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
clf = GaussianNB()

t0 = time()
clf.fit(features_train, labels_train)
print "Training Time: ", time()-t0, " secs"

t0 = time()
predicteds = clf.predict(features_test)
print "Prediction Time: ", time()-t0, " secs"

# Calc accuracy
print "Accuracy score: ", accuracy_score(labels_test, predicteds), " %"
#########################################################


