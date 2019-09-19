#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:32:20 2019

# Project's Content:
  Implement the skin detection classifier given in the lecture slides, using the Naïve Bayes classifier. The training
image and its ground truth are family.jpg and family.png. The test image and its ground truth are portrait.jpg and
portrait.png.
  Show the test image, its ground truth, and the detected binary mask in the same figure, as in the lecture slides.
  Calculate the true positive rate, true negative rate, false positive rate, and false negative rate for the test result.

@author: xiaoyuwang
"""
from PIL import Image
import numpy as np
import os
import math
import matplotlib.pyplot as plt

def open_image(image,image_type): 
    path_to_img = os.path.join('/Users/xiaoyuwang/Workspace/python/HW3/'+ str(image)+'.'+ str(image_type)) 
    return Image.open(path_to_img, 'r') 

def read_img(im):
    return list(im.getdata())

# calculate the mean value
def mean(num):
    return sum(num)/float(len(num))

# calculate the stdev, which is the square root of variance 
def stdev(num):
    avg = mean(num)
    variance = sum([pow(x-avg,2) for x in num])/float(len(num)-1)
    return math.sqrt(variance)

# calculate the Probability assuming Gaussian distributions
def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent

# creat the X = [x1, x2,....,xN], xk=[rk,gk]T
def create_matrix(x):
    r = np.zeros((len(x),1))
    g = np.zeros((len(x),1))   
    for i in range(0, len(x)):  
            R = x[i][0]
            G = x[i][1]
            B = x[i][2]
            if R+G+B == 0:
                r[i] = 1/3
                g[i] = 1/3
            else:
                r[i] = float(R)/(R+G+B)
                g[i] = float(G)/(R+G+B)
    X = np.column_stack((r,g)) 
    X = X.transpose()
    return X

# Calculate the conditional likelihood of xk 
def calculate_conditional_likelihood(train, test, k):
    x_mean = mean(train)
    x_stdev = stdev(train)
    prob = []
    for i in range(0,len(test[0])): 
        prob.append(calculateProbability(test[k][i], x_mean, x_stdev))
    return prob

# read the pixels of training and testing images
train_pixel = read_img(open_image('family','jpg'))
train_pixel_groundtrue = read_img(open_image('family','png'))
test_pixel = read_img(open_image('portrait','jpg'))
test_pixel_groundtrue = read_img(open_image('portrait','png'))
classification_result = test_pixel

# store the skin and background pixels of training images
skin = []
background = []   

# Classifier for Background and Skin
for i in range(0,len(train_pixel_groundtrue)):
        R = train_pixel_groundtrue[i][0]
        G = train_pixel_groundtrue[i][1]
        B = train_pixel_groundtrue[i][2]
        if R == 0 and G == 0 and B == 0:
            background.append(train_pixel[i])
        if R == 255 and G == 255 and B == 255:
            skin.append(train_pixel[i])
# calculate PH0/PH1
PH0_PH1 = float(len(background))/len(skin)

# store X separately for skin and background class
x_skin = create_matrix(skin)
x_back = create_matrix(background)

# Testing stage, Extract the feature vector Xk for the k-th pixel
test = create_matrix(test_pixel)

# Calculate the conditional likelihood of xk for testing stage
# P(xk|H1)
P_skin_r = calculate_conditional_likelihood(x_skin[0], test, 0)
P_skin_g = calculate_conditional_likelihood(x_skin[1], test, 1)
P_x_H1 = np.multiply(P_skin_r, P_skin_g)

# P(xk|H0)
P_back_r = calculate_conditional_likelihood(x_back[0], test, 0)
P_back_g = calculate_conditional_likelihood(x_back[1], test, 1)
P_x_H0 = np.multiply(P_back_r, P_back_g)

# calculate P(xk|H1)/P(xk|H0)
ratio_PH1_PH0 = P_x_H1/P_x_H0 

# store the skin/background labels of test image's ground true mask
test_skin_num = test_back_num = 0    
test_true_label = np.zeros((50000,1))
for i in range(0,len(test_pixel_groundtrue)):
        R = test_pixel_groundtrue[i][0]
        G = test_pixel_groundtrue[i][1]
        B = test_pixel_groundtrue[i][2]
        if R == 0 and G == 0 and B == 0: # background
            test_true_label[i] = 0 # H0: background
            test_back_num += 1
        if R == 255 and G == 255 and B == 255: # skin
            test_true_label[i] = 1 # H1: skin
            test_skin_num += 1

# Apply the classifier
label = np.zeros((50000,1))
for i in range(0,len(ratio_PH1_PH0)):
    if ratio_PH1_PH0[i] > PH0_PH1:
        label[i] = 1
        classification_result[i] = (255,255,255)
    if ratio_PH1_PH0[i] < PH0_PH1:
        label[i] = 0
        classification_result[i] = (0,0,0)

True_Pos_count = True_Neg_count = False_Pos_count = False_Neg_count = 0

for i in range(0,len(label)):
    if label[i] ==  test_true_label[i]: # True
        if label[i] == 1:
            True_Pos_count += 1 # True Positive
        else:
            True_Neg_count += 1 # True Negative
    if label[i] != test_true_label[i]: # False
        if label[i] == 1:
            False_Pos_count += 1   # False Positive
        else:
            False_Neg_count += 1   # False Negative

True_P_rate = (True_Pos_count/test_skin_num)*100
True_N_rate = (True_Neg_count/test_back_num)*100
False_P_rate = (False_Pos_count/test_back_num)*100       
False_N_rate = (False_Neg_count/test_skin_num)*100          

print('True Positive rate is:',True_P_rate,'%')
print('True Negative rate is:',True_N_rate,'%')
print('False Positive rate is:',False_P_rate,'%')
print('False Negative rate is:',False_N_rate,'%')


# Use PIL to create an image from the new array of pixels
test_image = open_image('portrait','jpg')
groundtruth_mask = open_image('portrait','png')
result_img = Image.new(test_image.mode, test_image.size)
result_img.putdata(classification_result)

# Show the Test image,Ground Truth Mask and Classification Result
plt.subplot(1,3,1)
plt.imshow(test_image)
plt.title('Test Image')
plt.xticks([])
plt.yticks([])

plt.subplot(1,3,2)
plt.imshow(groundtruth_mask)
plt.title('Ground Truth Mask')
plt.xticks([])
plt.yticks([])

plt.subplot(1,3,3)
plt.imshow(result_img)
plt.title('Classification Result')
plt.xticks([])
plt.yticks([])

plt.show()


