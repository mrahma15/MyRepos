#!/usr/bin/env python
# coding: utf-8

# In[13]:


# Project_SVR_With PCA 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[14]:


weather_solar_energy = pd.DataFrame(pd.read_csv("https://raw.githubusercontent.com/mrahma15/MyRepos/main/Hourly%20Weather%20and%20Solar%20Energy%20Dataset.csv"))
weather_solar_energy.head()


# In[15]:


m = len(weather_solar_energy)
m


# In[16]:


weather_solar_energy.shape


# In[17]:


num_vars = ['Cloud coverage', 'Visibility', 'Temperature', 'Dew point', 'Relative humidity', 'Wind speed', 'Station pressure', 'Altimeter', 'Solar energy']
data = housing[num_vars]

import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# define standard scaler
#scaler = StandardScaler()
scaler = MinMaxScaler()
data[num_vars] = scaler.fit_transform(data[num_vars])
data.head()


# In[18]:


Y = data.pop('Solar energy')
X = data


# In[19]:


X.head()


# In[20]:


Y.head()


# In[21]:


#importing Principal Component Analysis

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(X)
principalComponents = pd.DataFrame(principalComponents)
principalComponents


# In[22]:


#Setting the principal components as the input variables

X = principalComponents.iloc[:, [0, 1, 2, ]].values
X


# In[23]:


Y = Y.values

print('Y = ', Y[: 5])


# In[24]:


#importing train_test_split
#splitting the dataset into 80% and 20% split

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 100)


# In[25]:


# Linear Support vector regression

from sklearn.svm import SVR

svr_lin = SVR(kernel='linear', C=1e2)
svr_lin.fit(X_train, Y_train)
Y_pred = svr_lin.predict(X_test)


# In[26]:


# Final loss for linear suport vector regression

errors = np.subtract(Y_pred, Y_test)
sqrErrors = np.square(errors)
m_test = len(X_test)
loss = 1 / (2 * m_test) * np.sum(sqrErrors)
loss    


# In[27]:


#SVR with rbf kernel

svr_rbf = SVR(kernel='rbf', C=1e-2, gamma=0.1)
svr_rbf.fit(X_train, Y_train)
Y_pred = svr_rbf.predict(X_test)


# In[28]:


#Final loss for rbf kernel

errors = np.subtract(Y_pred, Y_test)
sqrErrors = np.square(errors)
m_test = len(X_test)
loss = 1 / (2 * m_test) * np.sum(sqrErrors)
loss 


# In[35]:


#SVR with polynomoial kernel

svr_poly = SVR(kernel='poly', C=1e3, degree=2)
svr_poly.fit(X_train, Y_train)
Y_pred = svr_poly.predict(X_test)


# In[36]:


#Final loss for polynomial kernel

errors = np.subtract(Y_pred, Y_test)
sqrErrors = np.square(errors)
m_test = len(X_test)
loss = 1 / (2 * m_test) * np.sum(sqrErrors)
loss 


# In[ ]:




