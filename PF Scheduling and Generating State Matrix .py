#!/usr/bin/env python
# coding: utf-8

# In[69]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[70]:


Num_UEs = 10
Bandwidth = 5   #MHz
Num_PRBs = 25
Num_Subframes = 5000


# In[71]:


Resource_Grid_np = np.zeros(shape=(Num_PRBs,Num_Subframes))
Resource_Grid = pd.DataFrame(Resource_Grid_np)
Resource_Grid


# In[72]:


Past_Avg_Throughput_np = np.random.seed(0)
Past_Avg_Throughput_np = np.random.rand(Num_UEs,Num_PRBs)
Past_Avg_Throughput = pd.DataFrame(Past_Avg_Throughput_np)
Past_Avg_Throughput


# In[73]:


SINR_perPRB_np = np.random.seed(42)
SINR_perPRB_np = np.random.rand(Num_UEs,Num_PRBs)
SINR_perPRB = pd.DataFrame(SINR_perPRB_np)
SINR_perPRB


# In[74]:


expected_datarate_perPRB_np = np.log2(1+SINR_perPRB)
expected_datarate_perPRB = pd.DataFrame(expected_datarate_perPRB_np)
expected_datarate_perPRB


# In[75]:


for t in range (0, Num_Subframes):
    Achieved_datarate_np = np.zeros(shape=(Num_UEs,Num_PRBs))
    Achieved_datarate = pd.DataFrame(Achieved_datarate_np)
    perRB_metric_np = np.divide(expected_datarate_perPRB, Past_Avg_Throughput)
    perRB_metric = pd.DataFrame(perRB_metric_np)
    UE_index_np = perRB_metric.columns.get_indexer(perRB_metric.apply('idxmax', axis=0))
    UE_index = pd.DataFrame(UE_index_np)
    Multiplier_mat_np = np.zeros(shape=(Num_UEs, Num_PRBs))
    Multiplier_mat = pd.DataFrame(Multiplier_mat_np)
    
    for j in range (0, Num_UEs):
        for k in range (0, Num_PRBs):
            if j == UE_index_np[k]:
                Multiplier_mat.iloc[[j],[k]] = 1
    Achieved_datarate = expected_datarate_perPRB * Multiplier_mat.values
    Past_Avg_Throughput = Past_Avg_Throughput + Achieved_datarate
    Resource_Grid.iloc[:,[t]] = UE_index
    
    
Resource_Grid


# In[76]:


Resource_Grid = Resource_Grid.astype(int)
Resource_Grid


# In[77]:


Resource_Grid = Resource_Grid.replace([0,8,2,1,7,3,4,5,6,9],[0,0,0,0,0,1,1,1,1,1])
Resource_Grid


# In[ ]:




