#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


weather_solar_energy = pd.DataFrame(pd.read_csv("https://raw.githubusercontent.com/mrahma15/MyRepos/main/Hourly%20Weather%20and%20Solar%20Energy%20Dataset.csv"))
weather_solar_energy.head()


# In[3]:


m = len(weather_solar_energy)
m


# In[4]:


weather_solar_energy.shape


# In[5]:


#Splitting the Data into Training and Testing Sets
from sklearn.model_selection import train_test_split

# We specify random seed so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(weather_solar_energy, train_size = 0.8, test_size = 0.2, random_state = 100)

df_train.shape


# In[6]:


df_test.shape


# In[7]:


num_vars = ['Cloud coverage', 'Visibility', 'Temperature', 'Dew point', 'Relative humidity', 'Wind speed', 'Station pressure', 'Altimeter', 'Solar energy']
df_Newtrain = df_train[num_vars]
df_Newtest = df_test[num_vars]
df_Newtrain.head()


# In[8]:


df_Newtrain.shape


# In[9]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# define standard scaler
#scaler = StandardScaler()
scaler = MinMaxScaler()
df_Newtrain[num_vars] = scaler.fit_transform(df_Newtrain[num_vars])
df_Newtrain.head(20)


# In[10]:


df_Newtest[num_vars] = scaler.fit_transform(df_Newtest[num_vars])
df_Newtest.head(20)


# In[11]:


y_Newtrain = df_Newtrain.pop('Solar energy')
X_Newtrain = df_Newtrain


# In[12]:


X_Newtrain.head()


# In[13]:


y_Newtrain.head()


# In[14]:


y = y_Newtrain.values

print('y = ', y[: 5])


# In[15]:


#importing Principal Component Analysis

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pc_train = pca.fit_transform(X_Newtrain)
pc_train = pd.DataFrame(pc_train)
pc_train


# In[17]:


# getting the input values from each column and putting them as a separate variable for training set

X1 = pc_train.values[:, 0]               
X2 = pc_train.values[:, 1]               
X3 = pc_train.values[:, 2]               
               
             
print('X1 = ', X1[: 5]) 
print('X2 = ', X2[: 5])
print('X3 = ', X3[: 5])


# In[27]:


m = len(pc_train)               # size of training set
X_0 = np.ones((m, 1))             # Creating a matrix of single column of ones as X0 with the size of training set
X_0 [: 5]


# In[28]:


# Converting 1D arrays of training X's to 2D arrays

X_1 = X1.reshape(m, 1)
X_2 = X2.reshape(m, 1)
X_3 = X3.reshape(m, 1)


print('X_1 = ', X_1[: 5])
print('X_2 = ', X_2[: 5])
print('X_3 = ', X_3[: 5])


# In[29]:


# Stacking X_0 through X_11 horizotally
# This is the final X Matrix for training

X = np.hstack((X_0, X_1, X_2, X_3))
X [: 5]


# In[30]:


theta = np.zeros(4)
theta


# In[22]:


y_Newtest = df_Newtest.pop('Solar energy')
X_Newtest = df_Newtest


# In[23]:


X_Newtest.head()


# In[24]:


y_Newtest.head()


# In[25]:


y_test = y_Newtest.values

print('y_test = ', y_test[: 5])


# In[26]:


pc_test = pca.fit_transform(X_Newtest)
pc_test = pd.DataFrame(pc_test)
pc_test


# In[31]:


# getting the input values from each column and putting them as a separate variable for validation set

X1_test = pc_test.values[:, 0]                
X2_test = pc_test.values[:, 1]                
X3_test = pc_test.values[:, 2]               
                


# In[32]:


m_test = len(pc_test)                # size of validation set
X_0_test = np.ones((m_test, 1))        # Creating a matrix of single column of ones as X0 with the size of validation set


# In[34]:


# Converting 1D arrays of validation X's to 2D arrays

X_1_test = X1_test.reshape(m_test, 1)
X_2_test = X2_test.reshape(m_test, 1)
X_3_test = X3_test.reshape(m_test, 1)


# In[35]:


# Stacking X_0_test through X_11_test horizotally
# This is the final X Matrix for validation

X_test = np.hstack((X_0_test, X_1_test, X_2_test, X_3_test))
X_test [: 5]


# In[36]:


# defining function for computing the cost 

def compute_cost(X, y, theta, m):
    predictions = X.dot(theta)
    errors = np.subtract(predictions, y)
    sqrErrors = np.square(errors)
    J = 1 / (2 * m) * np.sum(sqrErrors)
    return J


# In[37]:


# defining function for gradient descent algorithm
# gradient descent algorithm is applied on the training set
# for each iteration loss for both training and validation set is calculated

def gradient_descent(X, y, theta, alpha, iterations):
    cost_history = np.zeros(iterations)
    cost_test = np.zeros(iterations)
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = np.subtract(predictions, y)
        sum_delta = (alpha / m) * X.transpose().dot(errors);
        theta = theta - sum_delta;
        cost_history[i] = compute_cost(X, y, theta, m)                       # loss for training set
        cost_test[i] = compute_cost(X_test, y_test, theta, m_test)           # loss for training set
     
    return theta, cost_history, cost_test


# In[38]:


# computing the cost for initial theta values

cost = compute_cost(X, y, theta, m)
cost


# In[42]:


theta = [0., 0., 0., 0.]
iterations = 1000;
alpha = 0.1


# In[43]:


# Computing final theta values and losses for training and validation set

theta, cost_history, cost_test = gradient_descent(X, y, theta, alpha, iterations)
print('Final value of theta=', theta)
print('cost_history =', cost_history)
print('cost_test =', cost_test)


# In[44]:


plt.plot(range(1, iterations + 1),cost_history, color='blue', label= 'Loss for Training Set')
plt.plot(range(1, iterations + 1),cost_test, color='red', label= 'Loss for Evaluation Set')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.legend()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')


# In[ ]:




