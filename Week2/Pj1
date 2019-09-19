#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
## Project's Contents:
    Iris.xls contains 150 data samples of three Iris categories, labeled by outcome values 0, 1, and 2. Each
data sample has four attributes: sepal length, sepal width, petal length, and petal width.
Implement the K-means clustering algorithm to group the samples into K=3 clusters. Randomly choose three
samples as the initial cluster centers. Calculate the objective function value J as defined in Problem 3 after the
assignment step in each iteration. Exit the iterations if the following criterion is met: 𝐽(Iter − 1) − 𝐽(Iter) < ε,
where ε = 10−5, and Iter is the iteration number. Plot the objective function value J versus the iteration number
Iter.
Created on Mon Apr 15 09:52:31 2019
@author: xiaoyuwang
"""
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (8, 6)
plt.style.use('ggplot')

# Calculate euclidian distance
def dist(a, b, axis=1):
    return np.sqrt(np.sum((a-b)**2))

# Size of the K value
k = 3

# Load the excel file with pandas
DF = pd.read_excel('/Users/xiaoyuwang/Workspace/python/Iris.xls')

# All the data in one array
Iris = np.array(list(DF.iloc[0:150, 1:5].values))

# Pick random cluster centers
index1 = random.randint(0,150)
index2 = random.randint(0,150)
index3 = random.randint(0,150)
c1 = Iris[index1]
c2 = Iris[index2]
c3 = Iris[index3]

# combine 3 cluster centers into array
center = np.vstack((c1, c2, c3))

# Cluster
cluster = [[] for i in range(k)]

# Objective function array
J_array = []
for i in range(Iris.shape[0]):
    dp = Iris[i]
    distance = [dist(dp,center[0]),dist(dp,center[1]),dist(dp,center[2])]
    min = np.argmin(distance)
    cluster[min].append(dp)    

# Objective function value J
for i in range(k):    
    J = np.sum(dist(cluster[0],center[0]))+ np.sum(dist(cluster[1],center[1])) + np.sum(dist(cluster[2],center[2]))
J_array.append(J)

# Cluster-center update
for i in range(k):
    curCluster = cluster[i]
    sumCluster = sum(curCluster)
    avgCluster = sum(curCluster)/len(curCluster)
    center[i] = avgCluster

# Number of iteration
Iter = 0
Iter_array = []
Iter_array.append(Iter)

# Iteration
while (1):
    J_Old = J
    # Reset cluster
    cluster = [[] for i in range(k)]
    # Assignment step
    for i in range(Iris.shape[0]):
        dp = Iris[i]
        distance = [dist(dp,center[0]),dist(dp,center[1]),dist(dp,center[2])]
        min = np.argmin(distance)
        cluster[min].append(dp)    
    # Objective function value J
    for i in range(k):    
         J = np.sum(dist(cluster[0],center[0]))+ np.sum(dist(cluster[1],center[1])) + np.sum(dist(cluster[2],center[2]))
    J_array.append(J)
    # Cluster-center update
    for i in range(k):
        curCluster = cluster[i]
        sumCluster = sum(curCluster)
        avgCluster = sum(curCluster)/len(curCluster)
        center[i] = avgCluster
    Iter = Iter + 1
    Iter_array.append(Iter)
    # Exit the iteration
    if J_Old - J < 10**-5:
        break
print('J:',J_array)
print('Iteration number:', Iter_array)

## Plot objective value J & Iteration number Iter
plt.plot(Iter_array, J_array, '-o', c = 'blue')
plt.xlabel('Iteration number',fontsize = 16, color = 'black')
plt.ylabel('J',fontsize = 16,color = 'black')
