#!/usr/bin/env python
# coding: utf-8

# In[552]:


#Homework_04_Problem_01

#importing libraries for numpy, plot and pandas
#importing breast cancer dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer


# In[553]:


breast = load_breast_cancer()


# In[554]:


#loading the input variables into a matrix

breast_data = breast.data
breast_data.shape


# In[555]:


#converting the input variables matrix into a 2D panda array

breast_input = pd.DataFrame(breast_data)
breast_input.head()


# In[556]:


#loading the outputs into a matrix

breast_labels = breast.target
breast_labels.shape


# In[557]:


#reshaping the output row matrix into a column

labels = np.reshape(breast_labels,(569,1))


# In[558]:


#concatenating the input variables and outputs into the same 2D array to create the final dataset

final_breast_data = np.concatenate([breast_data,labels],axis=1)


# In[559]:


final_breast_data.shape


# In[560]:


#loading the feature names for the input variables

breast_dataset = pd.DataFrame(final_breast_data)
features = breast.feature_names
features


# In[561]:


features_labels = np.append(features,'label')


# In[562]:


breast_dataset.columns = features_labels


# In[563]:


breast_dataset.head()


# In[564]:


#breast_dataset['label'].replace(0, 'Benign',inplace=True)
#breast_dataset['label'].replace(1, 'Malignant',inplace=True)


# In[565]:


#breast_dataset.tail()


# In[566]:


#importing standard scalar and standarding the input values


from sklearn.preprocessing import StandardScaler

# Separating out the features
x = breast_dataset.loc[:, features].values
# Separating out the target
#y = breast_dataset.loc[:,['label']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)


# In[567]:


#importing Principal Component Analysis

from sklearn.decomposition import PCA
pca = PCA(n_components= 6)
principalComponents = pca.fit_transform(x)
principalComponents = pd.DataFrame(principalComponents)
principalComponents


# In[568]:


#Setting the principal components as the input variables

X = principalComponents.iloc[:, [0, 1, 2, 3, 4, 5]].values
Y = breast_dataset.iloc[:, 30].values


# In[569]:


X


# In[570]:


Y


# In[571]:


#importing train_test_split
#splitting the dataset into 80% and 20% split

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 100)


# In[572]:


# Importing Support Vector Machine
# Linear Support vector classifier

from sklearn.svm import SVC
classifier = SVC(kernel='linear', C=1E5)
classifier.fit(X_train, Y_train)


# In[573]:


#testing the classifier on the test data (Linear SVC)

Y_pred = classifier.predict(X_test)
Y_pred[0:9]


# In[574]:


#importing and creating confusion matrix (Linear SVC)

from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(Y_test, Y_pred)
cnf_matrix


# In[575]:


#importing and printing accuracy, precision and recall of the classifier (Linear SVC)

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Precision:",metrics.precision_score(Y_test, Y_pred))
print("Recall:",metrics.recall_score(Y_test, Y_pred))


# In[576]:


#plotting the confusion matrix (Linear SVC)

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


# In[577]:


# Support vector classifier with rbf kernel

classifier = SVC(kernel='rbf', C=1E5, gamma=0.1)
classifier.fit(X_train, Y_train)


# In[578]:


#testing the classifier on the test data (rbf kernel)

Y_pred = classifier.predict(X_test)
Y_pred[0:9]


# In[579]:


# creating confusion matrix (rbf kernel)

cnf_matrix = confusion_matrix(Y_test, Y_pred)
cnf_matrix


# In[580]:


# printing accuracy, precision and recall of the classifier (rbf kernel)

print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Precision:",metrics.precision_score(Y_test, Y_pred))
print("Recall:",metrics.recall_score(Y_test, Y_pred))


# In[581]:


#plotting the confusion matrix (rbf kernel)

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


# In[582]:


# Support vector classifier with polynomial kernel

classifier = SVC(kernel='poly', C=1E1, degree=3)
classifier.fit(X_train, Y_train)


# In[583]:


#testing the classifier on the test data (polynomial kernel)

Y_pred = classifier.predict(X_test)
Y_pred[0:9]


# In[584]:


# creating confusion matrix (polynomial kernel)

cnf_matrix = confusion_matrix(Y_test, Y_pred)
cnf_matrix


# In[585]:


# printing accuracy, precision and recall of the classifier (polynomial kernel)

print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Precision:",metrics.precision_score(Y_test, Y_pred))
print("Recall:",metrics.recall_score(Y_test, Y_pred))


# In[586]:


#plotting the confusion matrix (polynomial kernel)

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


# In[587]:


# Support vector classifier with sigmoid kernel

classifier = SVC(kernel='sigmoid', C=1)
classifier.fit(X_train, Y_train)


# In[588]:


#testing the classifier on the test data (sigmoid kernel)
 
Y_pred = classifier.predict(X_test)
Y_pred[0:9]


# In[589]:


# creating confusion matrix (sigmoid kernel)

cnf_matrix = confusion_matrix(Y_test, Y_pred)
cnf_matrix


# In[590]:


# printing accuracy, precision and recall of the classifier (sigmoid kernel)

print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Precision:",metrics.precision_score(Y_test, Y_pred))
print("Recall:",metrics.recall_score(Y_test, Y_pred))


# In[591]:


#plotting the confusion matrix (sigmoid kernel)

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

