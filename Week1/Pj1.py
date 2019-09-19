#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:12:00 2019
@author: xiaoyuwang
## Project's Contents:
The Pima Indians diabetes data set (pima-indians-diabetes.xlsx) is a data set used to diagnostically
predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset.
All patients here are females at least 21 years old of Pima Indian heritage. The dataset consists of M = 8 attributes
and one target variable, Outcome (1 represents diabetes, 0 represents no diabetes). The 8 attributes include
Pregnancies, Glucose, BloodPressure, BMI, insulin level, age, and so on. There are N=768 data samples.
Randomly select n samples from the “diabetes” class and n samples from the “no diabetes” class, and use them
as the training samples. The remaining data samples are the test samples. Build a linear regression model with the
training set, and test your model on the test samples to predict whether or not a test patient has diabetes or not.
Assume the predicted outcome of a test sample is 𝑡̂, if 𝑡̂ ≥ 0.5 (closer to 1), classify it as “diabetes”; if 𝑡̂ < 0.5
(closer to 0), classify it as “no diabetes”. Run 1000 independent experiments, and calculate the prediction accuracy
rate as 𝑡ℎ𝑒 𝑛𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝑐𝑜𝑟𝑟𝑒𝑐𝑡 𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛𝑠 / 𝑡ℎ𝑒 𝑡𝑜𝑡𝑎𝑙 𝑛𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝑡𝑒𝑠𝑡 𝑐𝑎𝑠𝑒𝑠 %. Let n=20, 40, 60, 80, 100, plot the accuracy rate versus n.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model 

# Loading the dataframe
DF = pd.read_excel('/Users/xiaoyuwang/Workspace/python/pima-indians-diabetes.xlsx')

# Defining the columns 
DF.columns =['Pregnancies', 'Glucose', 'Blood_pres', 'Skin_thick', 
             'Insulin', 'BMI', 'Diabetes_func', 'Age', 'Outcome']

# Split the data frame by 'Outcome'
df0 = DF[DF['Outcome'] == 0]
df1 = DF[DF['Outcome'] == 1]
print('Number of Data Sample with Outcome 0:', len(df0)) #500
print('Number of Data Sample with Outcome 1:', len(df1)) #268

# Randomly split the data into training and testing datasets
def split_train_test(df, m): # m is the test set size 
        X = np.array(df.drop(['Outcome'], axis = 1))
        y = np.array(df['Outcome'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = m)
        return X_train, X_test, y_train, y_test

# Calculate the correct detection count
def correct_detection(m1,m2): # m1,m2 is the test set size for Outcome=0 and Outcome=1
    X0_train, X0_test, y0_train, y0_test = split_train_test(df0, m1)
    X1_train, X1_test, y1_train, y1_test = split_train_test(df1, m2)
    X_train = np.concatenate([X0_train, X1_train])
    y_train = np.concatenate([y0_train, y1_train])
    X_test = np.concatenate([X0_test, X1_test])
    y_test = np.concatenate([y0_test, y1_test])
    # Apply the linear regression
    lm = linear_model.LinearRegression()
    # Train the data
    lm.fit(X_train,y_train)
    # Test the data
    y_pred = lm.predict(X_test)
    count = 0
    for j in range(len(y_test)):
        if (y_pred[j] >= 0.5 and y_test[j] == 1) or (y_pred[j] < 0.5 and y_test[j] == 0):
            count = count + 1
    return count

corrects = [0, 0, 0, 0, 0]   
N = [20, 40, 60, 80, 100]
pred_accuracy = [0, 0, 0, 0, 0]

# Run 1000 experiments
for i in range(1000):
    for j in range(len(N)):
        corrects[j] += correct_detection(len(df0) - N[j], len(df1) - N[j])
for j in range(len(N)):
    pred_accuracy[j] = float(corrects[j])/((len(df0) + len(df1) - 2 * N[j]) * 1000)
    
print('accuracy rate of the model is: ', pred_accuracy)
plt.plot(N, pred_accuracy, '-o')
plt.xlabel('n',fontsize = 16, color = 'black')
plt.ylabel('Accuracy Rate', fontsize = 16, color = 'black')
