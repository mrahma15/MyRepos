#!/usr/bin/env python
# coding: utf-8

# In[75]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[76]:


housing = pd.DataFrame(pd.read_csv("https://raw.githubusercontent.com/mrahma15/MyRepos/main/Housing.csv"))
housing.head()


# In[77]:


m = len(housing)
m


# In[78]:


housing.shape


# In[79]:


# List of variables to map
varlist = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# Defining the map function
def binary_map(x):
    return x.map({'yes': 1, "no": 0})

# Applying the function to the housing list
housing[varlist] = housing[varlist].apply(binary_map)

# Check the housing dataframe now
housing.head()


# In[80]:


#Splitting the Data into Training and Testing Sets
from sklearn.model_selection import train_test_split

# We specify random seed so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = 100)

df_train.shape


# In[81]:


df_test.shape


# In[82]:


num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'price']
df_Newtrain = df_train[num_vars]
df_Newtest = df_test[num_vars]
df_Newtrain.head()


# In[83]:


df_Newtrain.shape


# In[84]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# define standard scaler
#scaler = StandardScaler()
scaler = MinMaxScaler()
df_Newtrain[num_vars] = scaler.fit_transform(df_Newtrain[num_vars])
df_Newtrain.head(20)


# In[85]:


df_Newtest[num_vars] = scaler.fit_transform(df_Newtest[num_vars])
df_Newtest.head(20)


# In[86]:


y_Newtrain = df_Newtrain.pop('price')
X_Newtrain = df_Newtrain


# In[87]:


X_Newtrain.head()


# In[88]:


y_Newtrain.head()


# In[89]:


y = y_Newtrain.values

print('y = ', y[: 5])


# In[90]:


# getting the input values from each column and putting them as a separate variable for training set

X1 = df_Newtrain.values[:, 0]               
X2 = df_Newtrain.values[:, 1]              
X3 = df_Newtrain.values[:, 2]               
X4 = df_Newtrain.values[:, 3]               
X5 = df_Newtrain.values[:, 4]                
X6 = df_Newtrain.values[:, 5]                
X7 = df_Newtrain.values[:, 6]                
X8 = df_Newtrain.values[:, 7]                
X9 = df_Newtrain.values[:, 8]               
X10 = df_Newtrain.values[:, 9]              
X11 = df_Newtrain.values[:, 10]              

print('X1 = ', X1[: 5]) 
print('X2 = ', X2[: 5])
print('X3 = ', X3[: 5])
print('X4 = ', X4[: 5])
print('X5 = ', X5[: 5])
print('X6 = ', X6[: 5]) 
print('X7 = ', X7[: 5])
print('X8 = ', X8[: 5])
print('X9 = ', X9[: 5])
print('X10 = ', X10[: 5])
print('X11 = ', X11[: 5])


# In[91]:


m = len(X_Newtrain)               # size of training set
X_0 = np.ones((m, 1))             # Creating a matrix of single column of ones as X0 with the size of training set
X_0 [: 5]


# In[92]:


# Converting 1D arrays of training X's to 2D arrays

X_1 = X1.reshape(m, 1)
X_2 = X2.reshape(m, 1)
X_3 = X3.reshape(m, 1)
X_4 = X4.reshape(m, 1)
X_5 = X5.reshape(m, 1)
X_6 = X6.reshape(m, 1)
X_7 = X7.reshape(m, 1)
X_8 = X8.reshape(m, 1)
X_9 = X9.reshape(m, 1)
X_10 = X10.reshape(m, 1)
X_11 = X11.reshape(m, 1)

print('X_1 = ', X_1[: 5])
print('X_2 = ', X_2[: 5])
print('X_3 = ', X_3[: 5])
print('X_4 = ', X_4[: 5])
print('X_5 = ', X_5[: 5])
print('X_6 = ', X_6[: 5])
print('X_7 = ', X_7[: 5])
print('X_8 = ', X_8[: 5])
print('X_9 = ', X_9[: 5])
print('X_10 = ', X_10[: 5])
print('X_11 = ', X_11[: 5])


# In[93]:


# Stacking X_0 through X_11 horizotally
# This is the final X Matrix for training

X = np.hstack((X_0, X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8, X_9, X_10, X_11))
X [: 5]


# In[94]:


theta = np.zeros(12)
theta


# In[95]:


y_Newtest = df_Newtest.pop('price')
X_Newtest = df_Newtest


# In[96]:


X_Newtest.head()


# In[97]:


y_Newtest.head()


# In[98]:


y_test = y_Newtest.values

print('y_test = ', y_test[: 5])


# In[99]:


# getting the input values from each column and putting them as a separate variable for validation set

X1_test = df_Newtest.values[:, 0]                
X2_test = df_Newtest.values[:, 1]                
X3_test = df_Newtest.values[:, 2]                
X4_test = df_Newtest.values[:, 3]               
X5_test = df_Newtest.values[:, 4] 
X6_test = df_Newtest.values[:, 5]             
X7_test = df_Newtest.values[:, 6]                
X8_test = df_Newtest.values[:, 7]                
X9_test = df_Newtest.values[:, 8]               
X10_test = df_Newtest.values[:, 9] 
X11_test = df_Newtest.values[:, 10] 


# In[100]:


m_test = len(X_Newtest)                   # size of validation set
X_0_test = np.ones((m_test, 1))           # Creating a matrix of single column of ones as X0 with the size of validation set


# In[101]:


# Converting 1D arrays of validation X's to 2D arrays

X_1_test = X1_test.reshape(m_test, 1)
X_2_test = X2_test.reshape(m_test, 1)
X_3_test = X3_test.reshape(m_test, 1)
X_4_test = X4_test.reshape(m_test, 1)
X_5_test = X5_test.reshape(m_test, 1)
X_6_test = X6_test.reshape(m_test, 1)
X_7_test = X7_test.reshape(m_test, 1)
X_8_test = X8_test.reshape(m_test, 1)
X_9_test = X9_test.reshape(m_test, 1)
X_10_test = X10_test.reshape(m_test, 1)
X_11_test = X11_test.reshape(m_test, 1)


# In[102]:


# Stacking X_0_test through X_11_test horizotally
# This is the final X Matrix for validation

X_test = np.hstack((X_0_test, X_1_test, X_2_test, X_3_test, X_4_test, X_5_test, X_6_test, X_7_test, X_8_test, X_9_test, X_10_test, X_11_test))
X_test [: 5]


# In[103]:


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


# In[104]:


# defining function for computing the cost for validation set
# parameters are not penalized for validation set

def compute_cost_test(X_test, y_test, theta, m_test):
    predictions = X_test.dot(theta)
    errors = np.subtract(predictions, y_test)
    sqrErrors = np.square(errors)
    J = 1 / (2 * m_test) * np.sum(sqrErrors)
    return J


# In[105]:


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
        theta = np.array(theta)*(parameter_penalty) - sum_delta;                 # theta is penalized
        cost_history[i] = compute_cost(X, y, theta, m, Lambda)                   # loss for training set
        cost_test[i] = compute_cost_test(X_test, y_test, theta, m_test)          # loss for validation set
    
    return theta, cost_history, cost_test


# In[106]:


# to penalize theta values in gradient descent algorithm
# parameter_penalty has been introduced
# to exclude theta zero from penalizing, the first element of parameter_penalty is set to 1

theta = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
iterations = 1000;
alpha = 0.1
Lambda = 0.1
p = (1 - (alpha * Lambda) / m)
parameter_penalty = np.full(shape=11, fill_value=p)
parameter_penalty = np.insert(parameter_penalty, 0, 1)


# In[107]:


# computing the cost for initial theta values

cost = compute_cost(X, y, theta, m, Lambda)
cost


# In[108]:


# Computing final theta values and losses for training and validation set

theta, cost_history, cost_test = gradient_descent(X, y, theta, alpha, iterations, Lambda)
print('Final value of theta=', theta)
print('cost_history =', cost_history)
print('cost_test =', cost_test)


# In[109]:


plt.plot(range(1, iterations + 1),cost_history, color='blue', label= 'Loss for Training Set')
plt.plot(range(1, iterations + 1),cost_test, color='red', label= 'Loss for Evaluation Set')
plt.rcParams["figure.figsize"] = (10,6)
plt.grid()
plt.legend()
plt.xlabel('Number of iterations')
plt.ylabel('Cost (J)')
plt.title('Convergence of gradient descent')


# In[ ]:




