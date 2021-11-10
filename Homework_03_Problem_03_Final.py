#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Homework_03_Problem_03

#importing libraries for numpy, plot and pandas
#importing Linear Discriminant Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# In[2]:


#importing breast cancer dataset

from sklearn.datasets import load_breast_cancer
dt = load_breast_cancer()
X = dt.data
Y = dt.target


# In[3]:


#performing LDA feature extraction on the dataset

lda = LinearDiscriminantAnalysis(n_components=1)
lda_t = lda.fit_transform(X,Y)


# In[4]:


X


# In[5]:


Y


# In[6]:


#splitting the dataset using train_test_split

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, test_size = 0.2, random_state = 100)


# In[7]:


#importing standard scalar and standardizingng the input values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[8]:


#Importing naive bays and building the classifier

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)


# In[9]:


#evaluating the classifier using the evaluation set

Y_pred = classifier.predict(X_test)


# In[10]:


Y_pred[0:10]


# In[11]:


#importing metrics and reporting the accuracy, precision and recall

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Precision:",metrics.precision_score(Y_test, Y_pred))
print("Recall:",metrics.recall_score(Y_test, Y_pred))


# In[12]:


# creating confusion matrix

from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(Y_test, Y_pred)
cnf_matrix


# In[13]:


#plotting confusion matrix

import seaborn as sns
class_names=[0,1] # name of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[14]:


#LDA for classification:

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
lda.fit(X_train,y_train)
y_pred = lda.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[15]:


#confusion matrix for LDA Classification

cnf_mat = confusion_matrix(y_test,y_pred)
cnf_mat


# In[16]:


#plotting confusion matrix for LDA classification

import seaborn as sns
class_names=[0,1] # name of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_mat), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[ ]:




