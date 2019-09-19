
# -*- coding: utf-8 -*-
"""
## Proj's Content:
  You are given a face image database of 10 subjects. Each subject has 10 images of 112 × 92 pixels.
Use principal-component analysis (PCA) for dimensionality reduction. Find the subspaces of rank
K=1,2,3,6,10,20, 30, and 50 (i.e. find K principal components). For simplicity, for each subject, use face images
1,3,4,5,7,9 as the training images, and face images 2,6,8,10 as the test images. Convert each image to a vector of
length D=112 × 92 = 10304. Stack 6 training images of all 10 subjects to form a matrix of size 10304 × 60.
Apply PCA to this data matrix with different rank values. Project the face images to the rank-K subspace (i.e.
project the face images onto the K principal components) and apply the nearest-neighbor classifier in the subspace.
Plot the recognition accuracy rate (𝑛𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝑐𝑜𝑟𝑟𝑒𝑐𝑡 𝑐𝑙𝑎𝑠𝑠𝑖𝑓𝑖𝑐𝑎𝑡𝑖𝑜𝑛 /𝑡𝑜𝑡𝑎𝑙 𝑛𝑢𝑚𝑏𝑒𝑟 𝑜𝑓 𝑡𝑒𝑠𝑡 𝑐𝑎𝑠𝑒𝑠 %) versus different K values.

"""
from PIL import Image
from numpy import array
import numpy as np
import glob
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8, 6)
plt.style.use('ggplot')

# All PGM files' names 
all_file = [f for f in glob.glob("/Users/xiaoyuwang/Workspace/python/att_faces_10/*/*.pgm")]

# Read pgm files into array and Convert each image to a vector
def img_vec(n):
    arr = array(Image.open(all_file[n]))
    arr_single = arr.flatten()
    return arr_single

# For each subject, use face images 1,3,4,5,7,9 as the training images
# For subject 1, take all_file[0],[3],[4],[5],[7],[9] as training images
def train_stack(m):  
    face_train = np.column_stack((img_vec(m),img_vec(m+3), img_vec(m+4), img_vec(m+5), 
                               img_vec(m+7), img_vec(m+9)))
    return face_train

# For each subject, use face images 2,6,8,10 as the testing images
# For subject 1, take all_file[1],[2],[6],[8] as test images
def test_stack(p):
    face_test = np.column_stack((img_vec(p), img_vec(p+1), img_vec(p+5), img_vec(p+7)))
    return face_test

# apply PCA to the data matrix with different rank values K
def pca(K, TrainSet, TestSet):  
    pca = PCA(n_components = K)
    TrainSet_pca = pca.fit_transform(TrainSet) # Fit the model with trainset and apply the dimensionality reduction on train set
    TestSet_pca = pca.transform(TestSet) # Apply dimensionality reduction to test set.
    return TrainSet_pca, TestSet_pca

## Main Program
# Train set matrix of size 10304 × 60
i = 0
prev_face_Train = train_stack(0)
while(i < 90):
    i = i + 10
    face_TrainSet = np.column_stack((prev_face_Train, train_stack(i)))
    prev_face_Train = face_TrainSet   
face_TrainSet = face_TrainSet.T

# Test set matrix of size 10304 × 40
j = 1
prev_face_Test = test_stack(1)
while(j < 90):
    j = j + 10
    face_TestSet = np.column_stack((prev_face_Test, test_stack(j)))
    prev_face_Test = face_TestSet    
face_TestSet = face_TestSet.T

# Subject number of images
Sub_no = ['s1','s10', 's2', 's3', 's4','s5', 's6', 's7', 's8', 's9']
Sub_Train = np.repeat(Sub_no, 6)
Sub_Test = np.repeat(Sub_no, 4)

# Rank value K = 1,2,3,6,10,20, 30 and 50
K = [1, 2, 3, 6, 10, 20, 30, 50]
Accuracy_rate = []
for i in range(len(K)):
    TrainSet_pca, TestSet_pca = pca(K[i], face_TrainSet, face_TestSet)
    # apply the nearest-neighbor classifier in the subspace
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(TrainSet_pca, Sub_Train)
    test_pred = model.predict(TestSet_pca)
    count = 0
    for i in range(len(test_pred)):
        if test_pred[i] == Sub_Test[i]:
            count += 1
    accuracy_rate = count / face_TestSet.shape[0]*100
    Accuracy_rate.append(accuracy_rate)
print("K =", K)  
print("Recognition Accuracy Rate(%) =", Accuracy_rate)  
    
## Plot K & Correct rate
plt.plot(K, Accuracy_rate, '-o', c = 'blue')
plt.xlabel('K',fontsize = 16, color = 'black')
plt.ylabel('Recognition Accuracy Rate(%)',fontsize = 16,color = 'black')
