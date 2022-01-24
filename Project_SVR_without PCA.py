#!/usr/bin/env python
# coding: utf-8

# In[31]:


#Project_SVR_without PCA

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[32]:


weather_solar_energy = pd.DataFrame(pd.read_csv("https://raw.githubusercontent.com/mrahma15/MyRepos/main/Hourly%20Weather%20and%20Solar%20Energy%20Dataset.csv"))
weather_solar_energy.head()


# In[33]:


m = len(weather_solar_energy)
m


# In[34]:


weather_solar_energy.shape


# In[35]:


#Splitting the Data into Training and Testing Sets
from sklearn.model_selection import train_test_split

# We specify random seed so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = 100)

df_train.shape


# In[36]:


df_test.shape


# In[37]:


num_vars = ['Cloud coverage', 'Visibility', 'Temperature', 'Dew point', 'Relative humidity', 'Wind speed', 'Station pressure', 'Altimeter', 'Solar energy']
df_Newtrain = df_train[num_vars]
df_Newtest = df_test[num_vars]
df_Newtrain.head()


# In[38]:


df_Newtrain.shape


# In[39]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# define standard scaler
#scaler = StandardScaler()
scaler = MinMaxScaler()
df_Newtrain[num_vars] = scaler.fit_transform(df_Newtrain[num_vars])
df_Newtrain.head(20)


# In[40]:


df_Newtest[num_vars] = scaler.fit_transform(df_Newtest[num_vars])
df_Newtest.head(20)


# In[41]:


y_Newtrain = df_Newtrain.pop('Solar energy')
X_Newtrain = df_Newtrain


# In[42]:


X_Newtrain.head()


# In[43]:


y_Newtrain.head()


# In[44]:


y = y_Newtrain.values

print('y = ', y[: 5])


# In[45]:


# getting the input values from each column and putting them as a separate variable for training set

X1 = df_Newtrain.values[:, 0]               
X2 = df_Newtrain.values[:, 1]               
X3 = df_Newtrain.values[:, 2]               
X4 = df_Newtrain.values[:, 3]               
X5 = df_Newtrain.values[:, 4]               
X6 = df_Newtrain.values[:, 5]               
X7 = df_Newtrain.values[:, 6]                
X8 = df_Newtrain.values[:, 7]                
             

print('X1 = ', X1[: 5]) 
print('X2 = ', X2[: 5])
print('X3 = ', X3[: 5])
print('X4 = ', X4[: 5])
print('X5 = ', X5[: 5])
print('X6 = ', X6[: 5]) 
print('X7 = ', X7[: 5])
print('X8 = ', X8[: 5])


# In[46]:


m = len(X_Newtrain)               # size of training set
X_0 = np.ones((m, 1))             # Creating a matrix of single column of ones as X0 with the size of training set
X_0 [: 5]


# In[47]:


# Converting 1D arrays of training X's to 2D arrays

X_1 = X1.reshape(m, 1)
X_2 = X2.reshape(m, 1)
X_3 = X3.reshape(m, 1)
X_4 = X4.reshape(m, 1)
X_5 = X5.reshape(m, 1)
X_6 = X6.reshape(m, 1)
X_7 = X7.reshape(m, 1)
X_8 = X8.reshape(m, 1)


print('X_1 = ', X_1[: 5])
print('X_2 = ', X_2[: 5])
print('X_3 = ', X_3[: 5])
print('X_4 = ', X_4[: 5])
print('X_5 = ', X_5[: 5])
print('X_6 = ', X_6[: 5])
print('X_7 = ', X_7[: 5])
print('X_8 = ', X_8[: 5])


# In[48]:


# Stacking X_0 through X_11 horizotally
# This is the final X Matrix for training

X = np.hstack((X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8))
X [: 5]


# In[49]:


y_Newtest = df_Newtest.pop('Solar energy')
X_Newtest = df_Newtest


# In[50]:


X_Newtest.head()


# In[51]:


y_Newtest.head()


# In[52]:


y_test = y_Newtest.values

print('y_test = ', y_test[: 5])


# In[53]:


# getting the input values from each column and putting them as a separate variable for validation set

X1_test = df_Newtest.values[:, 0]                
X2_test = df_Newtest.values[:, 1]                
X3_test = df_Newtest.values[:, 2]               
X4_test = df_Newtest.values[:, 3]                
X5_test = df_Newtest.values[:, 4] 
X6_test = df_Newtest.values[:, 5]              
X7_test = df_Newtest.values[:, 6]                 
X8_test = df_Newtest.values[:, 7]                


# In[54]:


m_test = len(X_Newtest)                # size of validation set
X_0_test = np.ones((m_test, 1))        # Creating a matrix of single column of ones as X0 with the size of validation set


# In[55]:


# Converting 1D arrays of validation X's to 2D arrays

X_1_test = X1_test.reshape(m_test, 1)
X_2_test = X2_test.reshape(m_test, 1)
X_3_test = X3_test.reshape(m_test, 1)
X_4_test = X4_test.reshape(m_test, 1)
X_5_test = X5_test.reshape(m_test, 1)
X_6_test = X6_test.reshape(m_test, 1)
X_7_test = X7_test.reshape(m_test, 1)
X_8_test = X8_test.reshape(m_test, 1)


# In[56]:


# Stacking X_0_test through X_11_test horizotally
# This is the final X Matrix for validation

X_test = np.hstack((X_0_test, X_1_test, X_2_test, X_3_test, X_4_test, X_5_test, X_6_test, X_7_test, X_8_test))
X_test [: 5]


# In[57]:


# Linear Support vector regression

from sklearn.svm import SVR

svr_lin = SVR(kernel='linear', C=1e3)
svr_lin.fit(X, y)

y_pred = svr_lin.predict(X_test)


# In[58]:


# Final loss for linear suport vector regression

errors = np.subtract(y_pred, y_test)
sqrErrors = np.square(errors)
loss = 1 / (2 * m_test) * np.sum(sqrErrors)
loss    


# In[59]:


#SVR with rbf kernel

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(X, y)
y_pred = svr_rbf.predict(X_test)


# In[60]:


#Final loss for rbf kernel

errors = np.subtract(y_pred, y_test)
sqrErrors = np.square(errors)
loss = 1 / (2 * m_test) * np.sum(sqrErrors)
loss  


# In[61]:


#SVR with polynomoial kernel

svr_poly = SVR(kernel='poly', C=1e3, degree=2)
svr_poly.fit(X, y)
y_pred = svr_poly.predict(X_test)


# In[62]:


#Final loss for polynomial kernel

errors = np.subtract(y_pred, y_test)
sqrErrors = np.square(errors)
loss = 1 / (2 * m_test) * np.sum(sqrErrors)
loss  

