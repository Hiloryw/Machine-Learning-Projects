#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 22:00:24 2019

# Project's Content:
  Use both PCA and the Fisher’s Linear Discriminant (FLD) method to find the
projection directions to reduce the face image dimension, followed by a nearest-neighbor classifier to perform
face recognition. For FLD, first use PCA to reduce the dimensionality of face images to 𝑑0 = 40. The final
reduced dimension of the images is 𝑑 = [1,2,3,6,10,20,30].

@author: xiaoyuwang
"""
import numpy as np
import os
import cv2
import random
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt  

plt.rcParams['figure.figsize'] = (8, 6)
plt.style.use('ggplot')

train_faces_count = 8
test_faces_count = 2
faces_count = 10
m = 92
n = 112
mn = m*n
l = 8*faces_count
K_values = np.array([1,2,3,6,10,20,30]) # K principal components
K_values = K_values.transpose()

K1 = 40
sum_pca_rate = np.zeros((len(K_values),1))
sum_lda_rate = np.zeros((len(K_values),1))

num_exp = 20
for n_exp in range(num_exp):
    training_ids = random.sample([1,2,3,4,5,6,7,8,9,10],train_faces_count)
    testing_ids = np.setdiff1d([1,2,3,4,5,6,7,8,9,10],training_ids)
    y_type = range(1,faces_count+1)
    y_train_all = np.tile(y_type, train_faces_count)
    y_test_all = np.tile(y_type, test_faces_count)
    pca_rate = np.zeros((len(K_values),1))
    lda_rate = np.zeros((len(K_values),1))
    L = np.empty(shape=(mn, l), dtype='float64')  # each row of L represents one train image
    
    cur_img = 0 
    for training_id in training_ids:
        for face_id in range(1,faces_count+1):
            path_to_img = os.path.join('/Users/xiaoyuwang/Workspace/python/att_faces_10/', 's' + str(face_id), str(training_id) + '.pgm')
            img = cv2.imread(path_to_img, 0)  
            img_col = np.array(img, dtype='float64').flatten()
            L[:, cur_img] = img_col[:]   # set the cur_img-th column to the current training image
            cur_img += 1
    L = L.transpose()
    
    t1=0
    t2=0
    
    for K in K_values: 
        # method 2 :1st step of LDA, PCA for dime-reducion
        pca0 = PCA(n_components = K1)
        pca0_operator = pca0.fit(L)
        L0 = pca0_operator.transform(L)
        
        # method 1 direct PCA
        pca = PCA(n_components = K)
        pca_operator = pca.fit(L)
        train_pca = pca_operator.transform(L)#.transpose()
    
        # method 2 : 2nd step of LDA
        lda = LinearDiscriminantAnalysis(n_components = K)
        lda_operator = lda.fit(L0, y_train_all)
        train_lda = lda_operator.transform(L0)#.transpose()
        
        Test = np.empty(shape=(mn, 20), dtype='float64') 
        curr = 0
        pca_correct_rate = 0
        lda_correct_rate = 0
        
        for testing_id in testing_ids:
            for face_id in range(1,faces_count+1):
                path_to_img = os.path.join('/Users/xiaoyuwang/Workspace/python/att_faces_10/', 's' + str(face_id), str(testing_id) + '.pgm')    
                img = cv2.imread(path_to_img, 0)                                        # read it as a grayscale image
                img_col = np.array(img, dtype='float64').flatten() 
                Test[:, curr] = img_col[:]   
                curr += 1
        Test = Test.transpose()    
        
      ## nearest neighbor classifier      
        # method 1 direct PCA
        test_pca = pca_operator.transform(Test)
        KNN_pca = KNeighborsClassifier(n_neighbors=1)
        KNN_pca.fit(train_pca, y_train_all)
        test_pca_pred = KNN_pca.predict(test_pca) 
        for i in range(len(test_pca_pred)):
            if test_pca_pred[i] == y_test_all[i]:
                 pca_correct_rate += 1
        pca_correct_rate /= (faces_count*test_faces_count)
        pca_rate[t1,0] = pca_correct_rate*100
        t1 += 1
        
        # method 2 LDA
        test2 = pca0_operator.transform(Test)
        test_lda = lda_operator.transform(test2)#.transpose()    
        KNN_lda = KNeighborsClassifier(n_neighbors=1)
        KNN_lda.fit(train_lda, y_train_all)
        test_lda_pred = KNN_lda.predict(test_lda) 
        
        for i in range(len(test_lda_pred)):
            if test_lda_pred[i] == y_test_all[i]:
                 lda_correct_rate += 1
        lda_correct_rate /= (faces_count*test_faces_count)
        lda_rate[t2,0] = lda_correct_rate*100
        t2 += 1    
    
    sum_pca_rate += pca_rate
    sum_lda_rate += lda_rate
    
pca_accuracy_rate = sum_pca_rate/num_exp
lda_accuracy_rate = sum_lda_rate/num_exp
plt.plot(K_values,lda_accuracy_rate,'-o', c = 'red',label='LDA')
plt.plot(K_values,pca_accuracy_rate,'-o', c = 'blue',label='PCA')
plt.legend(loc='center right') 
plt.xlabel('K',fontsize = 16, color = 'black')
plt.ylabel('Classification Accuracy Rate (%)',fontsize = 16,color = 'black')
    
        
        
        
        
        
        
