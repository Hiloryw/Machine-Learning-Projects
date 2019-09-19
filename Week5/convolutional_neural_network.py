#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:49:25 2019
## Project's Contents:
  Build a convolutional neural network for the hand-written digits recognition task with the MNIST
data set. Use the cross-entropy error function, and run 5 epochs. Give the recognition accuracy rate and show the
confusion matrix, both for the test set.

@author: xiaoyuwang
"""
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), fontsize = 8,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) # the 1st 2d-convolutional layer
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1))) # the 2nd 2d-convolutional layer
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None))
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1))) # the 3rd 2d-convolutional layer
model.add(layers.Flatten()) 
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(train_images, train_labels, epochs = 1)

test_labels_hat = model.predict_classes(test_images) 

num_correct = 0

for i in range(len(test_labels)):
    if test_labels_hat[i]==test_labels[i]:
        num_correct += 1

Accuracy_rate = num_correct/len(test_labels)*100
print("Accuracy Rate = ", Accuracy_rate,"%")

confusion_matrix(test_labels, test_labels_hat, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
class_names = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plot_confusion_matrix(test_labels, test_labels_hat, classes=class_names,
                      title='Confusion matrix, without normalization')

plot_confusion_matrix(test_labels, test_labels_hat, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
