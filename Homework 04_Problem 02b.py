#!/usr/bin/env python
# coding: utf-8

# In[397]:


# Homework_04_Problem 2 (With PCA Feature Extraction)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[398]:


housing = pd.DataFrame(pd.read_csv("https://raw.githubusercontent.com/mrahma15/MyRepos/main/Housing.csv"))
housing.head()


# In[399]:


m = len(housing)
m


# In[400]:


housing.shape


# In[401]:


# List of variables to map
varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']

# Defining the map function
def binary_map(x):
    return x.map({'yes': 1, 'no': 0, 'unfurnished': 0, 'semi-furnished': 0.5, 'furnished': 1})

# Applying the function to the housing list
housing[varlist] = housing[varlist].apply(binary_map)

# Check the housing dataframe now
housing.head()


# In[402]:


num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'price']
data = housing[num_vars]

import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# define standard scaler
scaler = StandardScaler()
#scaler = MinMaxScaler()
data[num_vars] = scaler.fit_transform(data[num_vars])
data.head()


# In[403]:


Y = data.pop('price')
X = data


# In[404]:


X.head()


# In[405]:


Y.head()


# In[406]:


#importing Principal Component Analysis

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalComponents = pd.DataFrame(principalComponents)
principalComponents


# In[407]:


#Setting the principal components as the input variables

X = principalComponents.iloc[:, [0, 1]].values
X


# In[408]:


Y = Y.values

print('Y = ', Y[: 5])


# In[409]:


#importing train_test_split
#splitting the dataset into 80% and 20% split

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 100)


# In[410]:


# Linear Support vector regression

from sklearn.svm import SVR

svr_lin = SVR(kernel='linear', C=1e2)
svr_lin.fit(X_train, Y_train)
Y_pred = svr_lin.predict(X_test)


# In[411]:


# Final loss for linear suport vector regression

errors = np.subtract(Y_pred, Y_test)
sqrErrors = np.square(errors)
m_test = len(X_test)
loss = 1 / (2 * m_test) * np.sum(sqrErrors)
loss    


# In[412]:


#SVR with rbf kernel

svr_rbf = SVR(kernel='rbf', C=1e-2, gamma=0.1)
svr_rbf.fit(X_train, Y_train)
Y_pred = svr_rbf.predict(X_test)


# In[413]:


#Final loss for rbf kernel

errors = np.subtract(Y_pred, Y_test)
sqrErrors = np.square(errors)
m_test = len(X_test)
loss = 1 / (2 * m_test) * np.sum(sqrErrors)
loss 


# In[414]:


#SVR with polynomoial kernel

svr_poly = SVR(kernel='poly', C=1e4, degree=3)
svr_poly.fit(X_train, Y_train)
Y_pred = svr_poly.predict(X_test)


# In[415]:


#Final loss for polynomial kernel

errors = np.subtract(Y_pred, Y_test)
sqrErrors = np.square(errors)
m_test = len(X_test)
loss = 1 / (2 * m_test) * np.sum(sqrErrors)
loss 

