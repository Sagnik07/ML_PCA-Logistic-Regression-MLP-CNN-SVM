import numpy as np
import math
import pandas as pd
import random
import glob
import cv2
import os
from numpy import mean
from numpy import cov
from numpy.linalg import eig
from sklearn.preprocessing import MinMaxScaler
import sys

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def logistic_regression(X_train, y_train, iter, alpha):
  theta = np.zeros(X_train.shape[1])
  count = 1
  while(count<=iter):
    temp = np.dot(X_train, theta)
    h = sigmoid(temp)
    count = count+1

    error = h-y_train


    gradient = np.dot(X_train.T, error)/y_train.shape[0]
    theta = theta - alpha*gradient

  return theta


arg_list = sys.argv
train_path = arg_list[1]
test_path = arg_list[2]

train_file = open(train_path, "r")
train_lines = train_file.readlines()
img_files = []
img_labels = []
for line in train_lines:
    img,label = line.split(" ")
    # print("Img: ",img, " Label: ",label)
    img_files.append(img)
    img_labels.append(label.strip())
    
train_file.close()

img_labels = np.array(img_labels)
unique_labels = np.unique(img_labels)
# unique_labels

y_train = pd.get_dummies(img_labels)
y_train = np.array(y_train)
# print(y_train)
# print(y_train.shape)

faces = []
for f1 in img_files:
    img = cv2.imread(f1)
    img = rgb2gray(img)
    img = cv2.resize(img, (64,64), interpolation = cv2.INTER_AREA)
    face = img.flatten()
    faces.append(face) 

img_mean = mean(faces)
temp = faces - img_mean
covariance = cov(temp.T)

e, v = eig(covariance)
indexes = e.argsort()[::-1]   
eigen_values = e[indexes]
eigen_vectors = v[:,indexes]
# print(eigen_values)
# print(eigen_vectors)

total = np.sum(eigen_values)
variance_reqd = 0.90
var_list = list()
no_of_components = 0
components_reqd = 0
sum1 = 0
while True:
  sum1 += eigen_values[no_of_components]
  var = sum1/total
  var_list.append(var)
  var_achieved = np.real(var)
  if var_achieved >= variance_reqd:
      components_reqd = no_of_components + 1
      break
  no_of_components += 1


# print("No. of components required to achieve less than 10% error: ", components_reqd)

vectors = eigen_vectors[:,:components_reqd]
transformation_matrix = np.real(vectors)
pca_projections = np.dot(faces, transformation_matrix)
reconstruction_matrix = np.dot(pca_projections, transformation_matrix.T)

X_train = np.array(reconstruction_matrix)
# print(X_train)
# print(X_train.shape)

X_train1 = X_train
scaler = MinMaxScaler()
scaler.fit(X_train1)
X_train1=scaler.transform(X_train1)

ones = np.ones([X_train1.shape[0],1])
X_train1 = np.concatenate((ones,X_train1),axis=1)
# print(X_train1)
# print(X_train1.shape)

num_of_classes = y_train.shape[1]
weights = [[0 for i in range(X_train1.shape[1])]]
for i in range(num_of_classes):
  weights = np.vstack((weights,logistic_regression(X_train1, y_train[:,i], 10000, 0.01)))

weights = weights[1:]
# print(weights)
# print(weights.shape)

test_file = open(test_path, "r")
test_lines = test_file.readlines()
X_test = []
test_images = []
for line in test_lines:
    test_images.append(line.strip())
    img = cv2.imread(line.strip())
    img = rgb2gray(img)
    img = cv2.resize(img, (64,64), interpolation = cv2.INTER_AREA)
    face = img.flatten()
    X_test.append(face) 
    
test_file.close()
X_test = np.array(X_test)
scaler.fit(X_test)
X_test=scaler.transform(X_test)
ones = np.ones([X_test.shape[0],1])
X_test = np.concatenate((ones,X_test),axis=1)
# print(X_test)
# print(X_test.shape)

predictions = sigmoid(np.dot(X_test, weights.T))
# print(predictions)
# print(predictions.shape)

y_pred = []
for i in range(predictions.shape[0]):
  maxElement = np.amax(predictions[i])
  for j in range(8):
    if(predictions[i][j] == maxElement):
      ind = j
      break
  y_pred.append(j)

# y_pred
# j = 0
# y_true = []
# y_predicted = []
for i in y_pred:
    # str1 = test_images[j]
    # temp = str1.rindex('/')
    # labely = str1[temp+1:temp+4]
    # y_true.append(labely)
    # y_predicted.append(unique_labels[i])
    print(unique_labels[i])
    # j = j+1
    
#0 -> alice
#2 -> bob
#5 -> abc

# from sklearn.metrics import accuracy_score
# print(accuracy_score(y_true, y_predicted))