#!/usr/bin/env python
# coding: utf-8

# In[110]:


#importing libraries for numpy, plot and pandas

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[111]:


#reading the dataset

dataset = pd.read_csv('https://raw.githubusercontent.com/mrahma15/MyRepos/main/diabetes.csv')


# In[112]:


dataset.head()


# In[113]:


#Separating the inputs and output from the dataset

X = dataset.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values
Y = dataset.iloc[:, 8].values


# In[114]:


X[0:10]


# In[115]:


Y[0:10]


# In[116]:


#splitting the dataset using train_test_split

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size = 0.8, test_size = 0.2, random_state = 100)


# In[117]:


#scaling the input data using standardiztion

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[118]:


#Importing naive bays and building the classifier

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)


# In[119]:


#evaluating the classifier using the evaluation set

Y_pred = classifier.predict(X_test)


# In[120]:


Y_pred[0:10]


# In[121]:


#importing metrics and reporting the accuracy, precision and recall

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Precision:",metrics.precision_score(Y_test, Y_pred))
print("Recall:",metrics.recall_score(Y_test, Y_pred))


# In[122]:


# creating confusion matrix

from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(Y_test, Y_pred)
cnf_matrix


# In[123]:


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


# In[ ]:




