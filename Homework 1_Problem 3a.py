#!/usr/bin/env python
# coding: utf-8

# In[360]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[361]:


housing = pd.DataFrame(pd.read_csv("https://raw.githubusercontent.com/mrahma15/MyRepos/main/Housing.csv"))
housing.head()


# In[362]:


m = len(housing)
m


# In[363]:


housing.shape


# In[364]:


# List of variables to map
varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# Defining the map function
def binary_map(x):
    return x.map({'yes': 1, "no": 0})

# Applying the function to the housing list
housing[varlist] = housing[varlist].apply(binary_map)

# Check the housing dataframe now
housing.head()


# In[365]:


#Splitting the Data into Training and Testing Sets
from sklearn.model_selection import train_test_split

# We specify random seed so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = 100)

df_train.shape


# In[366]:


df_test.shape


# In[367]:


num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']
df_Newtrain = df_train[num_vars]
df_Newtest = df_test[num_vars]
df_Newtrain.head()


# In[368]:


df_Newtrain.shape


# In[369]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# define standard scaler
#scaler = StandardScaler()
scaler = MinMaxScaler()
df_Newtrain[num_vars] = scaler.fit_transform(df_Newtrain[num_vars])
df_Newtrain.head(20)


# In[370]:


df_Newtest[num_vars] = scaler.fit_transform(df_Newtest[num_vars])
df_Newtest.head(20)


# In[371]:


y_Newtrain = df_Newtrain.pop('price')
X_Newtrain = df_Newtrain


# In[372]:


X_Newtrain.head()


# In[373]:


y_Newtrain.head()


# In[374]:


y = y_Newtrain.values

print('y = ', y[: 5])


# In[375]:


# getting the input values from each column and putting them as a separate variable for training set

X1 = df_Newtrain.values[:, 0]                
X2 = df_Newtrain.values[:, 1]                 
X3 = df_Newtrain.values[:, 2]               
X4 = df_Newtrain.values[:, 3]                
X5 = df_Newtrain.values[:, 4]                

print('X1 = ', X1[: 5]) 
print('X2 = ', X2[: 5])
print('X3 = ', X3[: 5])
print('X4 = ', X4[: 5])
print('X5 = ', X5[: 5])


# In[376]:


m = len(X_Newtrain)               # size of training set
X_0 = np.ones((m, 1))             # Creating a matrix of single column of ones as X0 with the size of training set
X_0 [: 5]


# In[377]:


# Converting 1D arrays of training X's to 2D arrays

X_1 = X1.reshape(m, 1)
X_2 = X2.reshape(m, 1)
X_3 = X3.reshape(m, 1)
X_4 = X4.reshape(m, 1)
X_5 = X5.reshape(m, 1)

print('X_1 = ', X_1[: 5])
print('X_2 = ', X_2[: 5])
print('X_3 = ', X_3[: 5])
print('X_4 = ', X_4[: 5])
print('X_5 = ', X_5[: 5])


# In[378]:


# Stacking X_0 through X_5 horizotally
# This is the final X Matrix for training

X = np.hstack((X_0, X_1, X_2, X_3, X_4, X_5))
X [: 5]


# In[379]:


theta = np.zeros(6)
theta


# In[380]:


y_Newtest = df_Newtest.pop('price')
X_Newtest = df_Newtest


# In[381]:


X_Newtest.head()


# In[382]:


y_Newtest.head()


# In[383]:


y_test = y_Newtest.values

print('y_test = ', y_test[: 5])


# In[384]:


# getting the input values from each column and putting them as a separate variable for validation set

X1_test = df_Newtest.values[:, 0]               
X2_test = df_Newtest.values[:, 1]                 
X3_test = df_Newtest.values[:, 2]                
X4_test = df_Newtest.values[:, 3]                
X5_test = df_Newtest.values[:, 4] 


# In[385]:


m_test = len(X_Newtest)                   # size of validation set
X_0_test = np.ones((m_test, 1))           # Creating a matrix of single column of ones as X0 with the size of validation set


# In[386]:


# Converting 1D arrays of validation X's to 2D arrays

X_1_test = X1_test.reshape(m_test, 1)
X_2_test = X2_test.reshape(m_test, 1)
X_3_test = X3_test.reshape(m_test, 1)
X_4_test = X4_test.reshape(m_test, 1)
X_5_test = X5_test.reshape(m_test, 1)


# In[387]:


# Stacking X_0_test through X_5_test horizotally
# This is the final X Matrix for validation

X_test = np.hstack((X_0_test, X_1_test, X_2_test, X_3_test, X_4_test, X_5_test))
X_test [: 5]


# In[388]:


# defining function for computing the cost for training set
# parameter penalty is introduced in loss function
# theta zero has not been penalized

def compute_cost(X, y, theta, m, Lambda):
    predictions = X.dot(theta)
    errors = np.subtract(predictions, y)
    sqrErrors = np.square(errors)
    sqrTheta = np.square(theta)
    sqrTheta_new = np.delete(sqrTheta, 0)                                  # square of theta zero is excluded
    J = 1 / (2 * m) * (np.sum(sqrErrors) + Lambda * np.sum(sqrTheta_new)) 
    return J


# In[389]:


# defining function for computing the cost for validation set
# parameters are not penalized for validation set

def compute_cost_test(X_test, y_test, theta, m_test):
    predictions = X_test.dot(theta)
    errors = np.subtract(predictions, y_test)
    sqrErrors = np.square(errors)
    J = 1 / (2 * m_test) * np.sum(sqrErrors)
    return J


# In[390]:


# defining function for gradient descent algorithm
# gradient descent algorithm is applied on the training set
# gradient descent is modified for parameter penalty
# for each iteration loss for both training and validation set is calculated

def gradient_descent(X, y, theta, alpha, iterations, Lambda):
    cost_history = np.zeros(iterations)
    cost_test = np.zeros(iterations)
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = np.subtract(predictions, y)
        sum_delta = (alpha / m) * X.transpose().dot(errors);
        theta = np.array(theta)*(parameter_penalty) - sum_delta;                      # theta is penalized
        cost_history[i] = compute_cost(X, y, theta, m, Lambda)                        # loss for training set
        cost_test[i] = compute_cost_test(X_test, y_test, theta, m_test)               # loss for validation set
    
    return theta, cost_history, cost_test


# In[391]:


# to penalize theta values in gradient descent algorithm
# parameter_penalty has been introduced
# to exclude theta zero from penalizing, the first element of parameter_penalty is set to 1

theta = [0., 0., 0., 0., 0., 0.]
iterations = 500;
alpha = 0.1
Lambda = 0.1
p = (1 - (alpha * Lambda) / m)
parameter_penalty = np.full(shape=5, fill_value=p)
parameter_penalty = np.insert(parameter_penalty, 0, 1)


# In[392]:


# computing the cost for initial theta values

cost = compute_cost(X, y, theta, m, Lambda)
cost


# In[393]:


# Computing final theta values and losses for training and validation set

theta, cost_history, cost_test = gradient_descent(X, y, theta, alpha, iterations, Lambda)
print('Final value of theta=', theta)
print('cost_history =', cost_history)
print('cost_test =', cost_test)


# In[394]:


plt.plot(range(1, iterations + 1),cost_history, color='blue', label= 'Loss for Training Set')
plt.plot(range(1, iterations + 1),cost_test, color='red', label= 'Loss for Evaluation Set')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.legend()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')


# In[ ]:




