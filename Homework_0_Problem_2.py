#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[30]:


df = pd.read_csv('https://raw.githubusercontent.com/mrahma15/MyRepos/main/D3.csv')
X1 = df.values[:, 0]                # get input value from first column as Variable 1
X2 = df.values[:, 1]                # get input value from second column as Variable 2
X3 = df.values[:, 2]                # get input value from third column as Variable 3
y = df.values[:, 3]                 # get output value from fourth column
m = len(y)
print('X1 = ', X1[: 5]) 
print('X2 = ', X2[: 5])
print('X3 = ', X3[: 5])
print('y = ', y[: 5])
print('m = ', m)


# In[31]:


X_0 = np.ones((m, 1))             # Creating a matrix of single column of ones as X0
X_0[:5]


# In[32]:


# Converting 1D arrays of X1, X2 and X3 to 2D arrays

X_1 = X1.reshape(m, 1)
X_2 = X2.reshape(m, 1)
X_3 = X3.reshape(m, 1)
print('X_1 = ', X_1[: 5])
print('X_2 = ', X_2[: 5])
print('X_3 = ', X_3[: 5])


# In[33]:


# Stacking X_0, X_1, X_2 and X_3 horizotally
# This is the final X Matrix

X = np.hstack((X_0, X_1, X_2, X_3))
X [: 5]


# In[34]:


theta = np.zeros(4)
theta


# In[35]:


# defining function for computing the cost for linear regression

def compute_cost(X, y, theta):
    predictions = X.dot(theta)
    errors = np.subtract(predictions, y)
    sqrErrors = np.square(errors)
    J = 1 / (2 * m) * np.sum(sqrErrors)
    return J


# In[36]:


# defining function for gradient descent algorithm

def gradient_descent(X, y, theta, alpha, iterations):
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        predictions = X.dot(theta)
        errors = np.subtract(predictions, y)
        sum_delta = (alpha / m) * X.transpose().dot(errors);
        theta = theta - sum_delta;
        cost_history[i] = compute_cost(X, y, theta)
    
    return theta, cost_history


# In[37]:


# computing the cost for initial theta values

cost = compute_cost(X, y, theta)
cost


# In[38]:


theta = [0., 0., 0., 0.]
iterations = 1500;
alpha = 0.1


# In[39]:


# Computing final theta values and cost history

theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)
print('Final value of theta=', theta)
print('cost_history =', cost_history)


# In[40]:


plt.plot(range(1, iterations + 1),cost_history, color='blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')


# In[41]:


# Predicting outputs for new X1, X2 and X3 values

X_new = ([1, 1, 1, 1],
        [1, 2, 0, 4],
        [1, 3, 2, 1])

Predictions_new = np.dot(X_new, theta)
Predictions_new

