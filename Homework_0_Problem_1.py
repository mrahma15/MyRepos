#!/usr/bin/env python
# coding: utf-8

# In[48]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[49]:


df = pd.read_csv('https://raw.githubusercontent.com/mrahma15/MyRepos/main/D3.csv')
X1 = df.values[:, 0]                     # get input value from first column as Variable 1
X2 = df.values[:, 1]                     # get input value from second column as Variable 2
X3 = df.values[:, 2]                     # get input value from third column as Variable 3
y = df.values[:, 3]                      # get output value from fourth column
m = len(y)
print('X1 = ', X1[: 5]) 
print('X2 = ', X2[: 5])
print('X3 = ', X3[: 5])
print('y = ', y[: 5])
print('m = ', m)


# In[50]:


plt.scatter(X1,y, color='red',marker= '+')
plt.grid()
plt.rcParams["figure.figsize"] = (10,6)
plt.xlabel('Variable 1')
plt.ylabel('Output')
plt.title('Scatter plot of training data')


# In[51]:


plt.scatter(X2,y, color='red',marker= '+')
plt.grid()
plt.rcParams["figure.figsize"] = (10,6)
plt.xlabel('Variable 2')
plt.ylabel('Output')
plt.title('Scatter plot of training data')


# In[52]:


plt.scatter(X3,y, color='red',marker= '+')
plt.grid()
plt.rcParams["figure.figsize"] = (10,6)
plt.xlabel('Variable 3')
plt.ylabel('Output')
plt.title('Scatter plot of training data')


# In[53]:


X_0 = np.ones((m, 1))          # Creating a matrix of single column of ones as X0
X_0[:5]


# In[54]:


# Converting 1D arrays of X1, X2 and X3 to 2D arrays

X_1 = X1.reshape(m, 1)           
X_2 = X2.reshape(m, 1)
X_3 = X3.reshape(m, 1)
print('X_1 = ', X_1[: 5])
print('X_2 = ', X_2[: 5])
print('X_3 = ', X_3[: 5])


# In[55]:


# Stacking X_1, X_2, X_3 with X_0 horizotally (separately) where X_0 is the first column
# These are the final X1, X2, X3 matrices

X1 = np.hstack((X_0, X_1))
X2 = np.hstack((X_0, X_2))
X3 = np.hstack((X_0, X_3))
print('X1 = ', X1[: 5])
print('X2 = ', X2[: 5])
print('X3 = ', X3[: 5])


# In[56]:


theta = np.zeros(2)
theta


# In[57]:


# defining function for computing the cost for linear regression

def compute_cost(A, y, theta):
    predictions = A.dot(theta)
    errors = np.subtract(predictions, y)
    sqrErrors = np.square(errors)
    J = 1 / (2 * m) * np.sum(sqrErrors)
    return J


# In[58]:


# defining function for gradient descent algorithm

def gradient_descent(A, y, theta, alpha, iterations):
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        predictions = A.dot(theta)
        errors = np.subtract(predictions, y)
        sum_delta = (alpha / m) * A.transpose().dot(errors);
        theta = theta - sum_delta;
        cost_history[i] = compute_cost(A, y, theta)
    
    return theta, cost_history


# In[59]:


# computing the cost for each variable in isolation for initial theta values

cost_1 = compute_cost(X1, y, theta)
cost_2 = compute_cost(X2, y, theta)
cost_3 = compute_cost(X3, y, theta)
print('The cost for given values of theta and Variable 1 =', cost_1)
print('The cost for given values of theta and Variable 2 =', cost_2)
print('The cost for given values of theta and Variable 3 =', cost_3)


# In[60]:


theta = [0., 0.]
iterations = 1500;
alpha = 0.01


# In[61]:


# Computing final theta values and cost history for Variable 1

theta, cost_history = gradient_descent(X1, y, theta, alpha, iterations)
print('Final value of theta (Variable 1)=', theta)
print('cost_history (Variable 1) =', cost_history)


# In[62]:


plt.scatter(X1[:,1], y, color='red', marker= '+', label= 'Training Data')
plt.plot(X1[:,1],X1.dot(theta), color='green', label='Linear Regression')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Variable 1')
plt.ylabel('Output')
plt.title('Linear Regression Fit')
plt.legend()


# In[63]:


plt.plot(range(1, iterations + 1),cost_history, color='blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')


# In[64]:


theta, cost_history = gradient_descent(X2, y, theta, alpha, iterations)
print('Final value of theta (Variable 2)=', theta)
print('cost_history (Variable 2) =', cost_history)


# In[65]:


plt.scatter(X2[:,1], y, color='red', marker= '+', label= 'Training Data')
plt.plot(X2[:,1],X2.dot(theta), color='green', label='Linear Regression')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Variable 2')
plt.ylabel('Output')
plt.title('Linear Regression Fit')
plt.legend()


# In[66]:


plt.plot(range(1, iterations + 1),cost_history, color='blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')


# In[67]:


theta, cost_history = gradient_descent(X3, y, theta, alpha, iterations)
print('Final value of theta (Variable 3)=', theta)
print('cost_history (Variable 3) =', cost_history)


# In[68]:


plt.scatter(X3[:,1], y, color='red', marker= '+', label= 'Training Data')
plt.plot(X3[:,1],X3.dot(theta), color='green', label='Linear Regression')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Variable 3')
plt.ylabel('Output')
plt.title('Linear Regression Fit')
plt.legend()


# In[69]:


plt.plot(range(1, iterations + 1),cost_history, color='blue')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')

