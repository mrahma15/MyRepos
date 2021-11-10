#!/usr/bin/env python
# coding: utf-8

# In[159]:


#Homework_03_Problem_02

#importing libraries for numpy, plot and pandas
#importing breast cancer dataset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer


# In[160]:


breast = load_breast_cancer()


# In[161]:


#loading the input variables into a matrix

breast_data = breast.data
breast_data.shape


# In[162]:


#converting the input variables matrix into a 2D panda array

breast_input = pd.DataFrame(breast_data)
breast_input.head()


# In[163]:


#loading the outputs into a matrix

breast_labels = breast.target
breast_labels.shape


# In[164]:


#reshaping the output row matrix into a column

labels = np.reshape(breast_labels,(569,1))


# In[165]:


#concatenating the input variables and outputs into the same 2D array to create the final dataset

final_breast_data = np.concatenate([breast_data,labels],axis=1)


# In[166]:


final_breast_data.shape


# In[167]:


#loading the feature names for the input variables

breast_dataset = pd.DataFrame(final_breast_data)
features = breast.feature_names
features


# In[168]:


features_labels = np.append(features,'label')


# In[169]:


breast_dataset.columns = features_labels


# In[170]:


breast_dataset.head()


# In[171]:


#breast_dataset['label'].replace(0, 'Benign',inplace=True)
#breast_dataset['label'].replace(1, 'Malignant',inplace=True)


# In[172]:


#breast_dataset.tail()


# In[173]:


#importing standard scalar and standarding the input values


from sklearn.preprocessing import StandardScaler

# Separating out the features
x = breast_dataset.loc[:, features].values
# Separating out the target
#y = breast_dataset.loc[:,['label']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)


# In[174]:


#importing Principal Component Analysis

from sklearn.decomposition import PCA
pca = PCA(n_components=9)
principalComponents = pca.fit_transform(x)
principalComponents = pd.DataFrame(principalComponents)
principalComponents


# In[175]:


#finalDf = pd.concat([principalDf, breast_dataset[['label']]], axis = 1)


# In[176]:


#Setting the principal components as the input variables

X = principalComponents.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]].values
Y = breast_dataset.iloc[:, 30].values


# In[177]:


X


# In[178]:


Y


# In[179]:


#importing train_test_split
#splitting the dataset into 80% and 20% split

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 100)


# In[180]:


#Importing logistic regression

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, Y_train)


# In[181]:


#testing the classifier on the test data

Y_pred = classifier.predict(X_test)
Y_pred[0:9]


# In[182]:


#importing and creating confusion matrix

from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(Y_test, Y_pred)
cnf_matrix


# In[183]:


#importing and printing accuracy, precision and recall of the classifier

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Precision:",metrics.precision_score(Y_test, Y_pred))
print("Recall:",metrics.recall_score(Y_test, Y_pred))


# In[184]:


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

