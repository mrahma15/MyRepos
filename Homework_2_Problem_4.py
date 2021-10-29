#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries for numpy, plot and pandas

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


#reading the dataset

dataset = pd.read_csv('https://raw.githubusercontent.com/mrahma15/MyRepos/main/diabetes.csv')


# In[3]:


dataset.head()


# In[4]:


#Separating the inputs and output from the dataset

X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values
Y = dataset.iloc[:, 8].values


# In[5]:


X[0:10]


# In[6]:


Y[0:10]


# In[7]:


#importing Kfold cross validation and defining the value of K

k = 10
from sklearn.model_selection import KFold
kf = KFold(n_splits = k, random_state = None)


# In[8]:


#scaling the input data using standardiztion

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)


# In[9]:


#Importing naive bays classifier

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()


# In[10]:


from sklearn import metrics


# In[11]:


#splitting the dataset using Kfold and training the classifier using (K-1) folds and evaluating using the remaining fold
#process is repeated until each fold has been used for evaluation
#for each fold accuracy, precision and recall have been calculated and finally their average is taken

acc_score = []
prec_score = []
rec_score = []

for train_index , test_index in kf.split(X):
    X_train , X_test = X[train_index], X[test_index]
    Y_train , Y_test = Y[train_index], Y[test_index]
     
    classifier.fit(X_train,Y_train)
    Y_pred = classifier.predict(X_test)
     
    acc = metrics.accuracy_score(Y_pred , Y_test)
    acc_score.append(acc)
    
    prec = metrics.precision_score(Y_test, Y_pred)
    prec_score.append(prec)
    
    rec = metrics.recall_score(Y_test, Y_pred)
    rec_score.append(rec)
    
     
avg_acc_score = sum(acc_score)/k
avg_prec_score = sum(prec_score)/k
avg_rec_score = sum(rec_score)/k
 
print('Accuracy of each fold - {}'.format(acc_score))
print('Average accuracy : {}'.format(avg_acc_score))

print('Precision of each fold - {}'.format(prec_score))
print('Average precision : {}'.format(avg_prec_score))

print('Recall of each fold - {}'.format(rec_score))
print('Average recall : {}'.format(avg_rec_score))


# In[ ]:




