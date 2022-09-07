#!/usr/bin/env python
# coding: utf-8

# In[56]:


#Importing Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[57]:


#Number of UEs, PRBs and Subframes

Num_UEs = 3
Bandwidth = 5   #MHz
Num_PRBs = 5
Num_Subframes = 2000


# In[58]:


Set_UEs = list(range(1, Num_UEs+1))
Set_UEs


# In[59]:


#Initializing Resource Grid matrix with all zero values 

Resource_Grid_np = np.zeros(shape=(Num_PRBs,Num_Subframes))
Resource_Grid = pd.DataFrame(Resource_Grid_np)
Resource_Grid


# In[60]:


#Initializing Past average throughput of different UEs on different PRBs with random values

Past_Avg_Throughput_np = np.random.seed(0)
Past_Avg_Throughput_np = np.random.rand(Num_UEs,Num_PRBs)
Past_Avg_Throughput = pd.DataFrame(Past_Avg_Throughput_np)
Past_Avg_Throughput


# In[61]:


#Initializing SINR of different UEs on different PRBs with random values

SINR_perPRB_np = np.random.seed(42)
SINR_perPRB_np = np.random.rand(Num_UEs,Num_PRBs)
SINR_perPRB = pd.DataFrame(SINR_perPRB_np)
SINR_perPRB


# In[62]:


#Calculating expected datarate of different UEs on different PRBs from SINR

expected_datarate_perPRB_np = np.log2(1+SINR_perPRB)
expected_datarate_perPRB = pd.DataFrame(expected_datarate_perPRB_np)
expected_datarate_perPRB


# In[81]:


#Initializing achieved datarate of different UEs on different PRBs in the current subframe with all zero values

Achieved_datarate_np = np.zeros(shape=(Num_UEs,Num_PRBs))
Achieved_datarate = pd.DataFrame(Achieved_datarate_np)
#Achieved_datarate.iloc[[1],[2]] = expected_datarate_perPRB.iloc[[1],[2]]
Achieved_datarate


# In[82]:


#calculating per_RB metric of different UEs using the PF scheduling strategy in the current subframe

perRB_metric_np = np.divide(expected_datarate_perPRB, Past_Avg_Throughput)
perRB_metric = pd.DataFrame(perRB_metric_np)
perRB_metric


# In[83]:


#Finding the index number of the UE that has the highest metric on different PRBs

UE_index_np = perRB_metric.columns.get_indexer(perRB_metric.apply('idxmax', axis=0))
UE_index = pd.DataFrame(UE_index_np)
UE_index_np


# In[84]:


UE_index_np[1]


# In[85]:


#Initializing a multiplier matrix of size Num_UEs*Num_PRBs with all zero values

Multiplier_mat_np = np.zeros(shape=(Num_UEs, Num_PRBs))
Multiplier_mat = pd.DataFrame(Multiplier_mat_np)
Multiplier_mat


# In[86]:


#Updating the multiplier matrix by setting the corressponding PRBs' value to 1 which are to be allocated to a particular UE

for j in range (0, Num_UEs):
    for k in range (0, Num_PRBs):
        if j == UE_index_np[k]:
            Multiplier_mat.iloc[[j],[k]] = 1
Multiplier_mat


# In[87]:


#Calculating the achieved datarate of different UEs on different PRBs in the current Subframe

Achieved_datarate = expected_datarate_perPRB * Multiplier_mat.values
Achieved_datarate


# In[88]:


#Updating the Past average throughput for the next subframe by adding the achieved datarate in the current subframe

Past_Avg_Throughput = Past_Avg_Throughput + Achieved_datarate
Past_Avg_Throughput


# In[89]:


#Updating the Resource Grid allocation for the current subframe

Resource_Grid.iloc[:,[2]] = UE_index
Resource_Grid


# In[ ]:





# In[ ]:




