#!/usr/bin/env python
# coding: utf-8

# In[25]:


#Homework_03_Problem_01

#importing libraries for numpy, plot and pandas
#importing breast cancer dataset


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer


# In[26]:


breast = load_breast_cancer()


# In[27]:


#loading the input variables into a matrix

breast_data = breast.data
breast_data.shape


# In[28]:


#converting the input variables matrix into a 2D panda array

breast_input = pd.DataFrame(breast_data)
breast_input.head()


# In[29]:


#loading the outputs into a matrix

breast_labels = breast.target
breast_labels.shape


# In[30]:


#reshaping the output row matrix into a column

labels = np.reshape(breast_labels,(569,1))


# In[31]:


#concatenating the input variables and outputs into the same 2D array to create the final dataset

final_breast_data = np.concatenate([breast_data,labels],axis=1)


# In[32]:


final_breast_data.shape


# In[33]:


#loading the feature names for the input variables

breast_dataset = pd.DataFrame(final_breast_data)
features = breast.feature_names
features


# In[34]:


features_labels = np.append(features,'label')


# In[35]:


breast_dataset.columns = features_labels


# In[36]:


breast_dataset.head()


# In[37]:


#breast_dataset['label'].replace(0, 'Benign',inplace=True)
#breast_dataset['label'].replace(1, 'Malignant',inplace=True)


# In[38]:


#breast_dataset.tail()


# In[39]:


#Separating the inputs and output from the dataset

X = breast_dataset.iloc[:, [0, 1, 2 ,3 ,4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]].values
Y = breast_dataset.iloc[:, 30].values


# In[40]:


X


# In[41]:


Y


# In[42]:


#importing train_test_split
#splitting the dataset into 80% and 20% split

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 100)


# In[43]:


#importing standard scalar for standardization

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[44]:


#Importing logistic regression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)


# In[45]:


#testing the classifier on the test data

Y_pred = classifier.predict(X_test)
Y_pred[0:9]


# In[46]:


#importing and creating confusion matrix

from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(Y_test, Y_pred)
cnf_matrix


# In[47]:


#importing and printing accuracy, precision and recall of the classifier

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Precision:",metrics.precision_score(Y_test, Y_pred))
print("Recall:",metrics.recall_score(Y_test, Y_pred))


# In[48]:


#plotting the confusion matrix

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

