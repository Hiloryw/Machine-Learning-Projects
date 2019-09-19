
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 18:13:13 2019
## Project's Contents:
  The MNIST data set is consisted of gray-scale images of hand-written digits. Each image has 28×28 pixels. 
There are 10 classes, that is, digit 0, 1, 2, …, 9. There are 60,000 training images and 10,000 test images.
The goal is to recognize the digit on the image. Use multi-class logistic regression for the hand-written 
digit recognition task. Give the recognition accuracy rate, and show the confusion matrix.

@author: xiaoyuwang
"""
import numpy as np
import tensorflow as tf
mnist = tf.keras.datasets.mnist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt

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


(x_traino, y_train),(x_testo, y_test) = mnist.load_data()
x_train = np.reshape(x_traino,(60000,28*28))
x_test = np.reshape(x_testo,(10000,28*28))
x_train, x_test = x_train / 255.0, x_test / 255.0

logreg = LogisticRegression(solver='saga', multi_class='multinomial', max_iter = 100, verbose=2)
logreg.fit(x_train, y_train)

y_test_hat = logreg.predict(x_test)
num_correct = 0

for i in range(len(y_test)):
    if y_test_hat[i]==y_test[i]:
        num_correct += 1

Accuracy_rate = num_correct/len(y_test)*100
print("Accuracy Rate = ", Accuracy_rate,"%")

confusion_matrix(y_test, y_test_hat, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
class_names = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plot_confusion_matrix(y_test, y_test_hat, classes=class_names,
                      title='Confusion matrix, without normalization')

plot_confusion_matrix(y_test, y_test_hat, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()








